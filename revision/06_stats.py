"""成對統計檢定與效果量（EDC17 / R2C16 / R2C17 / R3C15 / R3C21 / R3C47）。

五個條件跑在完全相同的樣本上，所以比較必須是成對的。這裡做三件事：
  * 精確 McNemar 檢定（二項），並以 Holm 校正多重比較
  * paired bootstrap：對「樣本」重抽（不是對條件各自重抽），給 accuracy 與
    macro-F1 差值的 95% CI
  * 效果量：百分點差、相對誤差下降、兩系統預測相異的樣本數

本研究不微調權重，唯一隨機源是解碼；CI 建立在「樣本母體」上，這正是論文
實際要做的推論，因此單次執行 + item-level bootstrap 已足以回應 EDC17。
"""
import argparse
import itertools

import numpy as np
import pandas as pd

import common as C

# 依序相鄰比較 + 主要比較；(較弱條件, 較強條件, 這個比較在問什麼)
COMPARISONS = [
    ("C1_llama_only", "C2_rag_unopt_noaug", "加入 RAG 的效果"),
    ("C2_rag_unopt_noaug", "C3_rag_unopt_aug", "未最佳化 prompt 下，加入資料擴增的效果"),
    ("C2_rag_unopt_noaug", "C4_rag_opt_noaug", "未擴增下，加入 prompt 最佳化的效果"),
    ("C4_rag_opt_noaug", "C5_rag_opt_aug", "最佳化 prompt 下，加入資料擴增的效果（主要比較）"),
    ("C1_llama_only", "C5_rag_opt_aug", "完整流程相對於裸模型"),
]


def encode(labels, series):
    """把標籤轉成整數；invalid（None/NaN）一律為 -1，永遠不等於任何真實標籤。"""
    lut = {lab: i for i, lab in enumerate(labels)}
    return np.array([lut.get(x, -1) if not pd.isna(x) else -1 for x in series],
                    dtype=np.int16)


def macro_f1(y_true, y_pred, n_classes):
    """向量化 macro-F1，供 bootstrap 內迴圈使用（sklearn 在此會太慢）。"""
    f1s = np.empty(n_classes, dtype=np.float64)
    for c in range(n_classes):
        tp = np.count_nonzero((y_pred == c) & (y_true == c))
        fp = np.count_nonzero((y_pred == c) & (y_true != c))
        fn = np.count_nonzero((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        f1s[c] = (2 * tp / denom) if denom else 0.0
    return f1s.mean()


def mcnemar_exact(b, c):
    """精確 McNemar：在 H0 下 b ~ Binomial(b+c, 0.5) 的雙尾機率。"""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def holm(pvals):
    """Holm–Bonferroni 校正，回傳與輸入同序的校正後 p 值。"""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)          # 維持單調不遞減
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()
    out = C.run_dir(args.run)
    labels = C.VALID_LABELS

    frames = {}
    for p in sorted((out / "preds").glob("*.jsonl")):
        frames[p.stem] = pd.DataFrame(C.read_jsonl(p)).sort_values("item_id") \
                           .reset_index(drop=True)
    if not frames:
        C.die("preds/ 底下沒有資料")

    ref = next(iter(frames.values()))
    ids = ref["item_id"].tolist()
    for name, df in frames.items():
        if df["item_id"].tolist() != ids:
            C.die(f"{name} 的 item_id 序列與其他條件不同，無法成對比較")
    y_true = encode(labels, ref["true_label"])
    y_pred = {k: encode(labels, v["parsed_label"]) for k, v in frames.items()}
    n = len(ids)
    C.log(f"{len(frames)} 個條件 × {n} 筆成對樣本")

    # 每個條件的點估計
    point = {k: {"accuracy": float((v == y_true).mean()),
                 "macro_f1": float(macro_f1(y_true, v, len(labels))),
                 "n_invalid": int((v == -1).sum())}
             for k, v in y_pred.items()}
    for k, v in point.items():
        C.log(f"  {k:22} acc={v['accuracy']:.4f}  macroF1={v['macro_f1']:.4f}  "
              f"invalid={v['n_invalid']}")

    # 預先產生 bootstrap 的重抽索引：所有比較共用同一組，確保彼此可比
    rng = np.random.default_rng(C.SEED)
    boot_idx = rng.integers(0, n, size=(args.boot, n))

    raw_p, comparisons = [], []
    for lo, hi, question in COMPARISONS:
        if lo not in y_pred or hi not in y_pred:
            C.log(f"  略過 {lo} → {hi}（缺少預測檔）")
            continue
        a_ok = (y_pred[lo] == y_true)
        b_ok = (y_pred[hi] == y_true)
        # b = 只有 hi 答對；c = 只有 lo 答對
        b = int(np.count_nonzero(~a_ok & b_ok))
        c = int(np.count_nonzero(a_ok & ~b_ok))
        p = mcnemar_exact(b, c)
        raw_p.append(p)

        d_acc = np.empty(args.boot)
        d_f1 = np.empty(args.boot)
        for i in range(args.boot):
            sel = boot_idx[i]
            yt = y_true[sel]
            d_acc[i] = (y_pred[hi][sel] == yt).mean() - (y_pred[lo][sel] == yt).mean()
            d_f1[i] = (macro_f1(yt, y_pred[hi][sel], len(labels))
                       - macro_f1(yt, y_pred[lo][sel], len(labels)))

        err_lo = 1 - point[lo]["accuracy"]
        comparisons.append({
            "from": lo, "to": hi, "question": question,
            "mcnemar": {"b_only_to_correct": b, "c_only_from_correct": c,
                        "n_discordant": b + c, "exact_p": p},
            "delta_accuracy": point[hi]["accuracy"] - point[lo]["accuracy"],
            "delta_accuracy_pp": round(
                100 * (point[hi]["accuracy"] - point[lo]["accuracy"]), 3),
            "delta_accuracy_ci95": [float(np.percentile(d_acc, 2.5)),
                                    float(np.percentile(d_acc, 97.5))],
            "delta_macro_f1": point[hi]["macro_f1"] - point[lo]["macro_f1"],
            "delta_macro_f1_ci95": [float(np.percentile(d_f1, 2.5)),
                                    float(np.percentile(d_f1, 97.5))],
            "relative_error_reduction": (
                (point[hi]["accuracy"] - point[lo]["accuracy"]) / err_lo
                if err_lo > 0 else None),
            "n_samples_with_different_outcome": int(
                np.count_nonzero(y_pred[lo] != y_pred[hi])),   # R2C17
            "delta_invalid": point[hi]["n_invalid"] - point[lo]["n_invalid"],
        })

    for comp, adj in zip(comparisons, holm(np.array(raw_p))):
        comp["mcnemar"]["holm_adjusted_p"] = float(adj)
        ci = comp["delta_accuracy_ci95"]
        comp["ci_excludes_zero"] = bool(ci[0] > 0 or ci[1] < 0)
        C.log(f"\n  {comp['from']} → {comp['to']}")
        C.log(f"    Δacc={comp['delta_accuracy_pp']:+.2f}pp  "
              f"CI95=[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}]pp  "
              f"b={comp['mcnemar']['b_only_to_correct']} "
              f"c={comp['mcnemar']['c_only_from_correct']}  "
              f"p={comp['mcnemar']['exact_p']:.4g}  "
              f"Holm={adj:.4g}  CI 排除 0：{comp['ci_excludes_zero']}")

    C.save_json(out / "stats.json", {
        "n_paired_samples": n,
        "bootstrap": {"resamples": args.boot, "seed": C.SEED,
                      "scheme": "對樣本成對重抽（同一組索引套用到所有條件）"},
        "significance_test": "exact McNemar (binomial, two-sided)",
        "multiplicity_correction": "Holm",
        "point_estimates": point,
        "comparisons": comparisons,
        "note": ("accuracy 採慣例 A（invalid 視為錯誤，分母 = 全部樣本）。"
                 "本研究不微調權重，CI 建立在樣本母體上。"),
    })
    C.log(f"\n輸出完成 -> {out / 'stats.json'}")


if __name__ == "__main__":
    main()
