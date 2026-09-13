"""由結果檔生成所有表與圖 —— 任何數字都不得手打。

對應題目：
  R3C20 五條件整合表 / R3C21 各元件邊際貢獻 / R3C22 每個指標附不確定性
  R3C23 implementation settings 表 / R2C19+R3C24 dataset composition 表
  R3C26+R3C69 row-normalized 且同色階的混淆矩陣
  R3C27 最佳化軌跡改用完整 0–1 y 軸
  R3C25 worked example / R3C62 誤判案例 / R3C63+EDC19 cost 表
"""
import argparse

import numpy as np
import pandas as pd

import common as C

COND_LABEL = {
    "C1_llama_only": "Llama 3.1 only",
    "C2_rag_unopt_noaug": "+ RAG",
    "C3_rag_unopt_aug": "+ RAG + augmentation",
    "C4_rag_opt_noaug": "+ RAG + optimized prompt",
    "C5_rag_opt_aug": "+ RAG + optimized prompt + augmentation",
}


def write_table(df, path, caption):
    df.to_csv(path.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    md = [f"**{caption}**", "", df.to_markdown(index=False)]
    path.with_suffix(".md").write_text("\n".join(md), encoding="utf-8")
    C.log(f"  {path.stem}: {len(df)} 列")


def consolidated_table(metrics, stats, out):
    """R3C20 + R3C22：五條件並列，每個指標都帶不確定性。"""
    ci = {c["to"]: c for c in stats["comparisons"]} if stats else {}
    rows = []
    for cond in COND_LABEL:
        r = metrics["results"].get(cond)
        if not r:
            continue
        a = r["convention_A_primary"]
        d = ci.get(cond, {})
        rows.append({
            "Condition": COND_LABEL[cond],
            "Accuracy": round(a["accuracy"], 4),
            "Macro-F1": a["macro"]["f1"],
            "Weighted-F1": a["weighted"]["f1"],
            "Invalid (n)": r["n_invalid"],
            "Invalid (%)": round(100 * r["invalid_rate"], 2),
            "N": a["denominator"],
            "Δ Accuracy vs prev (pp)": d.get("delta_accuracy_pp"),
            "95% CI of Δ (pp)": (
                f"[{d['delta_accuracy_ci95'][0]*100:+.2f}, "
                f"{d['delta_accuracy_ci95'][1]*100:+.2f}]"
                if d.get("delta_accuracy_ci95") else None),
            "Holm p": (round(d["mcnemar"]["holm_adjusted_p"], 4)
                       if d.get("mcnemar") else None),
        })
    write_table(pd.DataFrame(rows), out / "tables" / "table_main_results",
                "Consolidated results across the five conditions "
                "(accuracy counts invalid responses as incorrect)")


def deltas_table(stats, out):
    """R3C21：每個元件的邊際貢獻，含 CI、p 值與相異樣本數。"""
    rows = []
    for c in stats["comparisons"]:
        lo, hi = c["delta_accuracy_ci95"]
        rows.append({
            "Comparison": f"{COND_LABEL.get(c['from'], c['from'])} → "
                          f"{COND_LABEL.get(c['to'], c['to'])}",
            "What it isolates": c["question"],
            "Δ Accuracy (pp)": c["delta_accuracy_pp"],
            "95% CI (pp)": f"[{lo*100:+.2f}, {hi*100:+.2f}]",
            "Δ Macro-F1": round(c["delta_macro_f1"], 4),
            "Relative error reduction": (
                round(c["relative_error_reduction"], 4)
                if c.get("relative_error_reduction") is not None else None),
            "Discordant (b/c)": f"{c['mcnemar']['b_only_to_correct']}/"
                                f"{c['mcnemar']['c_only_from_correct']}",
            "N differing": c["n_samples_with_different_outcome"],
            "McNemar p": round(c["mcnemar"]["exact_p"], 5),
            "Holm p": round(c["mcnemar"]["holm_adjusted_p"], 5),
            "Δ Invalid": c["delta_invalid"],
        })
    write_table(pd.DataFrame(rows), out / "tables" / "table_component_deltas",
                "Marginal contribution of each component")


def implementation_table(manifest, out):
    """R3C23：一張表把可重現性需要的設定全列出來。"""
    g, e, r, llm = (manifest["generation"], manifest["embedding"],
                    manifest["retrieval"], manifest["llm"])
    items = [
        ("Model", llm.get("model_tag")),
        ("Model digest", llm.get("model_digest")),
        ("Quantization", llm.get("quantization")),
        ("Parameters", llm.get("parameter_count")),
        ("Runtime", f"Ollama {llm.get('ollama_version')}"),
        ("Context length (num_ctx)", g.get("num_ctx")),
        ("Temperature", g.get("temperature")),
        ("Max output tokens (num_predict)", g.get("num_predict")),
        ("top_p / top_k", f"{g.get('top_p')} / {g.get('top_k')}"),
        ("Decoding seed", g.get("seed")),
        ("Embedding model", e.get("model_id")),
        ("Embedding dimension", e.get("dimension")),
        ("Embedding max_seq_length", e.get("max_seq_length")),
        ("Embeddings normalised", e.get("normalize_embeddings")),
        ("Retrieval index", r.get("noaug", {}).get("index_type")),
        ("Distance metric", e.get("distance_metric")),
        ("Corpus size (no aug / aug)",
         f"{r.get('noaug', {}).get('ntotal')} / {r.get('aug', {}).get('ntotal')}"),
        ("Retrieved neighbours (k)", r.get("top_k")),
        ("Similarity threshold", r.get("similarity_threshold_note")),
        ("Document format", r.get("document_format")),
        ("GPU", manifest["hardware"].get("gpu")),
        ("Python", (manifest["hardware"].get("python") or "").split()[0]),
    ]
    df = pd.DataFrame(items, columns=["Setting", "Value"])
    write_table(df, out / "tables" / "table_implementation_settings",
                "Implementation settings used to produce the reported results")


def dataset_table(rep, out):
    """R2C19 + R2C13 + R2C25 + R3C24：兩張表，涵蓋排除類別前後與評估集建立過程。"""
    # (1) 從原始 7 類到 train/test 切分
    chain = rep.get("provenance_chain") or {}
    rows = []
    for stage, d in chain.items():
        if not isinstance(d, dict) or "per_class" not in d:
            continue
        rows.append({"Stage": stage, "N": d.get("n", sum(d["per_class"].values())),
                     **d["per_class"],
                     "Note": d.get("reason") or d.get("method") or ""})
    if rows:
        df = pd.DataFrame(rows).fillna(0)
        for c in df.columns:
            if c not in ("Stage", "Note"):
                df[c] = df[c].astype(int)
        write_table(df, out / "tables" / "table_dataset_provenance",
                    "Class distribution from the original seven-class dataset "
                    "through class exclusion, cleaning, balancing and splitting")

    # (2) 評估集的建立：逐階段剔除
    rows = []
    for stage, d in rep["stages"].items():
        rows.append({"Stage": stage, "N": d["n"], **d.get("per_class", {}),
                     "Reason": d.get("reason", "")})
    df = pd.DataFrame(rows).fillna(0)
    for c in df.columns:
        if c not in ("Stage", "Reason"):
            df[c] = df[c].astype(int)
    write_table(df, out / "tables" / "table_dataset_composition",
                "Construction of the uncontaminated evaluation set "
                "(language: English; labels are repository-provided)")


def cost_table(metrics, baselines, out):
    """R3C63 + EDC19：推論成本與 baseline 的訓練成本並列。"""
    rows = []
    for cond, r in metrics["results"].items():
        lat = r.get("latency", {})
        rows.append({
            "System": COND_LABEL.get(cond, cond),
            "Type": "LLM (no training)",
            "Train cost": "none (no weight updates)",
            "Median latency (ms/query)": (round(lat["median_total_duration_ms"], 1)
                                          if lat.get("median_total_duration_ms") else None),
            "p95 latency (ms)": (round(lat["p95_total_duration_ms"], 1)
                                 if lat.get("p95_total_duration_ms") else None),
            "Mean prompt tokens": (round(lat["mean_prompt_tokens"], 1)
                                   if lat.get("mean_prompt_tokens") else None),
            "Total GPU sec": (round(lat["total_gpu_sec"], 1)
                              if lat.get("total_gpu_sec") else None),
        })
    if baselines:
        for name, b in baselines.get("baselines", {}).items():
            rows.append({
                "System": name,
                "Type": "supervised baseline",
                "Train cost": f"{b['train_sec']:.1f}s",
                "Median latency (ms/query)": round(
                    1000 * b["infer_sec"] / max(1, baselines["n_test"]), 2),
                "p95 latency (ms)": None,
                "Mean prompt tokens": None,
                "Total GPU sec": round(b["train_sec"] + b["infer_sec"], 1),
            })
    write_table(pd.DataFrame(rows), out / "tables" / "table_cost_runtime",
                "Inference cost and runtime on the hardware reported in the "
                "implementation settings table")


def confusion_figure(metrics, out):
    """R3C26 + R3C69：row-normalized、同色階、Predicted 在 x 軸。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conds = [c for c in COND_LABEL if c in metrics["confusion_matrices"]]
    if not conds:
        return
    labels = metrics["labels"]
    cols = labels + ["INVALID"]
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 4.2),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        cm = pd.DataFrame(metrics["confusion_matrices"][cond]).reindex(
            index=labels, columns=cols).fillna(0).to_numpy(dtype=float)
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)   # 同一色階
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True" if cond == conds[0] else "")
        ax.set_title(COND_LABEL[cond], fontsize=9)
        for i in range(norm.shape[0]):
            for j in range(norm.shape[1]):
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8,
                 label="Row-normalised proportion")
    p = out / "figures" / "fig_confusion_matrices.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    C.log(f"  fig_confusion_matrices.png（{len(conds)} 個面板，色階 0–1）")


def search_figure(out):
    """R3C27：最佳化軌跡使用完整 0–1 y 軸，不截斷。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for f in ("prompt_search_log.jsonl", "prompt_search_log.part2.jsonl"):
        if (out / f).exists():
            rows += C.read_jsonl(out / f)
    if not rows:
        return
    df = pd.DataFrame(rows)
    acc = (df[df["round"] == "round1"].groupby("template_idx")["correct"].mean()
           if "round1" in set(df["round"]) else df.groupby("template_idx")["correct"].mean())
    if acc.empty:
        return
    order = acc.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.bar(range(len(order)), order.to_numpy(), width=1.0)
    ax.set_ylim(0, 1)                     # 完整範圍，不截斷
    ax.set_xlabel(f"Candidate prompt (n={len(order)}, ranked)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Prompt candidates on the validation set "
                 f"(spread = {order.max()-order.min():.3f})")
    ax.axhline(order.max(), ls="--", lw=1,
               label=f"best {order.max():.3f}")
    ax.axhline(order.min(), ls=":", lw=1, label=f"worst {order.min():.3f}")
    ax.legend()
    p = out / "figures" / "fig_prompt_search.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200)
    plt.close(fig)
    C.log(f"  fig_prompt_search.png（y 軸 0–1，最佳 {order.max():.3f} / 最差 {order.min():.3f}）")


def qualitative_examples(out, n_each=3):
    """R3C25 worked example + R3C62 誤判案例。"""
    path = out / "preds" / "C5_rag_opt_aug.jsonl"
    if not path.exists():
        return
    df = pd.DataFrame(C.read_jsonl(path))
    test = pd.read_csv(out / "test_clean.csv").set_index("item_id")["statement"]
    df["statement"] = df["item_id"].map(test)
    df["correct"] = df["parsed_label"] == df["true_label"]

    picks = pd.concat([
        df[df["correct"]].head(n_each).assign(kind="retrieval helped"),
        df[~df["correct"] & df["parsed_label"].notna()].head(n_each)
          .assign(kind="misclassified"),
        df[df["parsed_label"].isna()].head(n_each).assign(kind="invalid response"),
    ])
    cols = ["kind", "item_id", "statement", "retrieved_source_keys",
            "true_label", "parsed_label", "invalid_reason", "raw_response"]
    write_table(picks[cols].reset_index(drop=True),
                out / "tables" / "table_qualitative_examples",
                "Worked examples: input, retrieved context, prediction, "
                "and error analysis")


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    args = ap.parse_args()
    out = C.run_dir(args.run)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    load = lambda n: C.load_json(out / n) if (out / n).exists() else None
    metrics, stats = load("metrics.json"), load("stats.json")
    manifest, rep = load("run_manifest.json"), load("exclusion_report.json")
    baselines = load("baselines.json")

    C.log("產生表格：")
    if rep:
        dataset_table(rep, out)
    if manifest:
        implementation_table(manifest, out)
    if metrics and stats:
        consolidated_table(metrics, stats, out)
        deltas_table(stats, out)
    if metrics:
        cost_table(metrics, baselines, out)
        C.log("產生圖：")
        confusion_figure(metrics, out)
    search_figure(out)
    qualitative_examples(out)

    C.log(f"\n輸出完成 -> {out / 'tables'} 與 {out / 'figures'}")


if __name__ == "__main__":
    main()
