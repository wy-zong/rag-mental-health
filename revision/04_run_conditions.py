"""在未接觸過的 test_clean 上執行五個實驗條件，逐筆留下預測紀錄。

逐筆 JSONL 是本次最重要的產物 —— 上一輪就是因為沒有保存它，
EDC17 / R2C16 要求的 paired test 才變成數學上不可能補算。

同時做兩件審稿人明確索取的事：
  * 記錄每筆檢索到的來源，讓「test 文本不會被檢索到」成為可驗證的斷言（R3C64）
  * 記錄 ollama 回傳的計時欄位，供 cost/latency 表使用（R3C63 / EDC19）
"""
import argparse
import time
from collections import Counter

import pandas as pd

import common as C

# (條件代號, 使用哪個索引, 使用哪個 prompt)
CONDITIONS = [
    ("C1_llama_only", None, "llama_only"),
    ("C2_rag_unopt_noaug", "noaug", "unoptimized"),
    ("C3_rag_unopt_aug", "aug", "unoptimized"),
    ("C4_rag_opt_noaug", "noaug", "optimized"),
    ("C5_rag_opt_aug", "aug", "optimized"),
]


def load_rag(which, cache):
    if which is None:
        return None
    if which not in cache:
        paths = {"noaug": (C.INDEX_NOAUG, C.DOCS_NOAUG),
                 "aug": (C.INDEX_AUG, C.DOCS_AUG)}[which]
        cache[which] = C.RagIndex(*paths)
    return cache[which]


def run_condition(name, rag, template, rows, manifest, model, opts, out_path,
                  test_keys, top_k, seed=None):
    """跑完一個條件並寫出 JSONL；回傳統計摘要。"""
    options = dict(opts)
    if seed is not None:
        options["seed"] = seed

    leaks = 0
    reasons = Counter()
    correct = 0
    t0 = time.time()
    with C.JsonlWriter(out_path) as w:
        for i, r in enumerate(rows):
            if rag is not None:
                ctx, src_keys = rag.retrieve_context(r["statement"], k=top_k)
            else:
                ctx, src_keys = "", []
            # R3C64：檢索到的來源不得落在評估集內
            hit = [k for k in src_keys if k in test_keys]
            leaks += len(hit)

            prompt = C.build_prompt(template, r["statement"], ctx)
            res = C.ollama_chat(prompt, model, options)
            label, reason, mode = C.parse_label(res["raw_response"])
            if res["error"]:
                label, reason = None, "api_error"
            if reason:
                reasons[reason] += 1
            correct += int(label == r["status"])

            w.write({
                "condition": name,
                "item_id": int(r["item_id"]),
                "statement_sha": C.sha256_text(str(r["statement"]))[:16],
                "true_label": r["status"],
                "raw_response": res["raw_response"],
                "parsed_label": label,
                "is_invalid": label is None,
                "invalid_reason": reason,
                "parse_mode": mode,
                "retrieved_source_keys": src_keys,
                "retrieval_leak": bool(hit),
                "prompt_eval_count": res.get("prompt_eval_count"),
                "eval_count": res.get("eval_count"),
                "total_duration_ns": res.get("total_duration_ns"),
                "wall_ns": res.get("wall_ns"),
                "error": res.get("error"),
                "model_digest": manifest["llm"].get("model_digest"),
                "seed": options.get("seed"),
            })
            if (i + 1) % 200 == 0:
                C.log(f"    {i+1}/{len(rows)}  acc={correct/(i+1):.4f}  "
                      f"{time.time()-t0:.0f}s")

    return {
        "n": len(rows),
        "accuracy_invalid_as_wrong": correct / len(rows),
        "n_invalid": sum(reasons.values()),
        "invalid_reasons": dict(reasons),
        "retrieval_leaks": leaks,
        "elapsed_sec": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--only", nargs="*", help="只跑指定條件（除錯用）")
    ap.add_argument("--limit", type=int, help="只跑前 N 筆（除錯用，正式執行不可使用）")
    ap.add_argument("--determinism-n", type=int, default=200,
                    help="determinism 檢查的樣本數（R3C14）")
    ap.add_argument("--determinism-repeats", type=int, default=3)
    args = ap.parse_args()
    out = C.run_dir(args.run)

    manifest = C.load_json(out / "run_manifest.json")
    prompts = C.load_json(out / "prompts.json")
    if not prompts.get("optimized"):
        C.die("prompts.json 尚無 optimized —— 請先執行 03_select_prompt.py")

    test = pd.read_csv(out / "test_clean.csv")
    if args.limit:
        test = test.head(args.limit)
        C.log(f"!! --limit={args.limit}，僅供除錯")
    rows = test.to_dict("records")
    test_keys = set(test["statement"].map(C.norm_key))
    opts = C.ollama_options(manifest)
    C.log(f"評估集 {len(rows)} 筆；解碼參數 {opts}")

    rag_cache = {}
    preds_dir = out / "preds"
    summary = {}

    for name, which, prompt_key in CONDITIONS:
        if args.only and name not in args.only:
            continue
        path = preds_dir / f"{name}.jsonl"
        if path.exists() and len(C.read_jsonl(path)) == len(rows):
            C.log(f"[{name}] 已完成，略過（{path.name}）")
            continue
        C.log(f"\n[{name}] 索引={which or '無'}  prompt={prompt_key}")
        stats = run_condition(
            name, load_rag(which, rag_cache), prompts[prompt_key], rows,
            manifest, args.model, opts, path, test_keys, args.top_k)
        summary[name] = stats
        C.log(f"  acc={stats['accuracy_invalid_as_wrong']:.4f}  "
              f"invalid={stats['n_invalid']}  洩漏={stats['retrieval_leaks']}  "
              f"{stats['elapsed_sec']/60:.1f} 分鐘")
        if stats["retrieval_leaks"]:
            C.die(f"{name} 檢索到評估集內的文本 {stats['retrieval_leaks']} 次 —— "
                  "評估集建立有誤，請重跑 01_build_eval_set.py")

    # ---------- determinism 檢查（R3C14 / R3C46） ----------
    # 本研究不微調權重，唯一的隨機來源是解碼；用少量重複量測一致率，
    # 即可回答「是否 deterministic」，不必付出多 seed 全面重跑的代價。
    det_rows = rows[:args.determinism_n]
    det_name, det_which, det_prompt = CONDITIONS[-1]
    C.log(f"\n[determinism] {det_name} × {args.determinism_repeats} 次 × {len(det_rows)} 筆")
    runs = []
    for rep in range(args.determinism_repeats):
        p = out / "determinism" / f"rep{rep}.jsonl"
        if not p.exists():
            run_condition(det_name, load_rag(det_which, rag_cache),
                          prompts[det_prompt], det_rows, manifest, args.model,
                          opts, p, test_keys, args.top_k)
        runs.append([r["parsed_label"] for r in C.read_jsonl(p)])

    agree = sum(1 for vals in zip(*runs) if len(set(vals)) == 1)
    det = {
        "condition": det_name,
        "n_items": len(det_rows),
        "repeats": args.determinism_repeats,
        "exact_agreement_rate": agree / len(det_rows) if det_rows else None,
        "seed": opts.get("seed"),
        "temperature": opts.get("temperature"),
        "interpretation": (
            "一致率為 1.0 代表在固定 seed 與本硬體上解碼是 deterministic；"
            "低於 0.99 則需在論文中說明殘餘變異來源，並考慮多 seed 重跑。"),
    }
    C.log(f"  逐筆一致率 = {det['exact_agreement_rate']:.4f}")

    C.save_json(out / "conditions_summary.json", {
        "conditions": summary,
        "determinism": det,
        "test_clean_n": len(rows),
        "top_k": args.top_k,
        "note": "正式指標由 05_metrics.py 從 preds/*.jsonl 重新計算；此處僅為執行摘要。",
    })
    C.log(f"\n輸出完成 -> {out}")


if __name__ == "__main__":
    main()
