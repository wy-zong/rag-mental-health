"""計算五個條件的指標，兩種 invalid 分母慣例並陳（EDC18 / R2C23 / R3C18）。

審稿人抓到的問題是「不同條件似乎用了不同分母」。這裡把兩種慣例都算出來、
都寫進 JSON，並在每個數字旁註明分母，讓讀者不必自己推。

  慣例 A（主要）：invalid 視為錯誤，分母 = 全部評估樣本數
  慣例 B（次要）：排除 invalid，分母 = 有效樣本數
"""
import argparse

import numpy as np
import pandas as pd

import common as C

INVALID_TOKEN = "INVALID"


def load_preds(out):
    """讀入所有條件的逐筆預測，並確認 item_id 集合完全一致。"""
    frames = {}
    for p in sorted((out / "preds").glob("*.jsonl")):
        df = pd.DataFrame(C.read_jsonl(p)).sort_values("item_id").reset_index(drop=True)
        frames[p.stem] = df
    if not frames:
        C.die("preds/ 底下沒有任何 .jsonl —— 請先執行 04_run_conditions.py")

    id_sets = {k: set(v["item_id"]) for k, v in frames.items()}
    ref_name, ref_ids = next(iter(id_sets.items()))
    for name, ids in id_sets.items():
        if ids != ref_ids:
            C.die(f"{name} 與 {ref_name} 的 item_id 集合不同，無法做 paired 比較")
    return frames


def metrics_one(y_true, y_pred_raw, labels):
    """回傳 (慣例A, 慣例B) 兩組指標。"""
    from sklearn.metrics import precision_recall_fscore_support

    y_true = np.asarray(y_true, dtype=object)
    # invalid 以一個不在 labels 內的 token 表示：sklearn 會把它算成該真實類別的
    # FN，且不計入任何類別的 FP —— 這正是慣例 A 想要的行為。
    y_a = np.array([p if p is not None else INVALID_TOKEN for p in y_pred_raw], dtype=object)
    valid_mask = np.array([p is not None for p in y_pred_raw])

    def bundle(yt, yp, denom, note):
        p, r, f, s = precision_recall_fscore_support(
            yt, yp, labels=labels, zero_division=0)
        mp, mr, mf, _ = precision_recall_fscore_support(
            yt, yp, labels=labels, average="macro", zero_division=0)
        wp, wr, wf, _ = precision_recall_fscore_support(
            yt, yp, labels=labels, average="weighted", zero_division=0)
        return {
            "denominator": int(denom),
            "denominator_note": note,
            "accuracy": float((yt == yp).sum() / denom) if denom else None,
            "per_class": {
                lab: {"precision": round(float(p[i]), 4),
                      "recall": round(float(r[i]), 4),
                      "f1": round(float(f[i]), 4),
                      "support": int(s[i])}
                for i, lab in enumerate(labels)
            },
            "macro": {"precision": round(float(mp), 4), "recall": round(float(mr), 4),
                      "f1": round(float(mf), 4)},
            "weighted": {"precision": round(float(wp), 4), "recall": round(float(wr), 4),
                         "f1": round(float(wf), 4)},
        }

    conv_a = bundle(y_true, y_a, len(y_true), "invalid 視為錯誤，分母 = 全部樣本")
    conv_b = bundle(y_true[valid_mask], y_a[valid_mask], int(valid_mask.sum()),
                    "排除 invalid，分母 = 有效樣本")
    return conv_a, conv_b


def confusion(y_true, y_pred_raw, labels):
    """含 INVALID 欄的混淆矩陣（列＝真實、行＝預測），供繪圖使用。"""
    cols = labels + [INVALID_TOKEN]
    m = pd.DataFrame(0, index=labels, columns=cols, dtype=int)
    for t, p in zip(y_true, y_pred_raw):
        m.loc[t, p if p is not None else INVALID_TOKEN] += 1
    return m


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    args = ap.parse_args()
    out = C.run_dir(args.run)

    manifest = C.load_json(out / "run_manifest.json")
    frames = load_preds(out)
    labels = C.VALID_LABELS
    C.log(f"載入 {len(frames)} 個條件，各 {len(next(iter(frames.values())))} 筆")

    results, matrices = {}, {}
    cm_dir = out / "confusion"
    cm_dir.mkdir(parents=True, exist_ok=True)
    for name, df in frames.items():
        preds = [None if pd.isna(x) else x for x in df["parsed_label"]]
        conv_a, conv_b = metrics_one(df["true_label"].tolist(), preds, labels)
        n_invalid = int(df["is_invalid"].sum())
        results[name] = {
            "n": len(df),
            "n_invalid": n_invalid,
            "invalid_rate": round(n_invalid / len(df), 4),
            "invalid_reasons": df.loc[df["is_invalid"], "invalid_reason"]
                                 .value_counts().to_dict(),
            "convention_A_primary": conv_a,
            "convention_B_secondary": conv_b,
            "latency": {
                "median_total_duration_ms": float(
                    np.median(df["total_duration_ns"].dropna()) / 1e6)
                if df["total_duration_ns"].notna().any() else None,
                "p95_total_duration_ms": float(
                    np.percentile(df["total_duration_ns"].dropna(), 95) / 1e6)
                if df["total_duration_ns"].notna().any() else None,
                "mean_prompt_tokens": float(df["prompt_eval_count"].dropna().mean())
                if df["prompt_eval_count"].notna().any() else None,
                "mean_output_tokens": float(df["eval_count"].dropna().mean())
                if df["eval_count"].notna().any() else None,
                "total_gpu_sec": float(df["total_duration_ns"].dropna().sum() / 1e9)
                if df["total_duration_ns"].notna().any() else None,
            },
        }
        cm = confusion(df["true_label"], preds, labels)
        matrices[name] = cm.to_dict()
        cm.to_csv(cm_dir / f"{name}.csv", encoding="utf-8")

        C.log(f"  {name:22} accA={conv_a['accuracy']:.4f} (n={conv_a['denominator']})  "
              f"accB={conv_b['accuracy']:.4f} (n={conv_b['denominator']})  "
              f"invalid={n_invalid}  macroF1_A={conv_a['macro']['f1']:.4f}")

    # num_ctx 截斷檢查：prompt token 數若逼近上限，代表內容被靜默截掉
    ctx = manifest["generation"]["num_ctx"]
    worst = max((f["prompt_eval_count"].max() for f in frames.values()
                 if f["prompt_eval_count"].notna().any()), default=None)
    ctx_ok = worst is None or worst < ctx

    C.save_json(out / "metrics.json", {
        "labels": labels,
        "invalid_definition": manifest["metrics_definitions"]["invalid_response"],
        "conventions": {
            "A_primary": manifest["metrics_definitions"]["primary_convention"],
            "B_secondary": manifest["metrics_definitions"]["secondary_convention"],
        },
        "results": results,
        "confusion_matrices": matrices,
        "num_ctx_check": {"num_ctx": ctx, "max_prompt_eval_count":
                          int(worst) if worst is not None else None, "ok": bool(ctx_ok)},
    })
    if not ctx_ok:
        C.log(f"!! prompt token 數 {worst} 已達 num_ctx={ctx}，內容可能被截斷")
    C.log(f"\n輸出完成 -> {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
