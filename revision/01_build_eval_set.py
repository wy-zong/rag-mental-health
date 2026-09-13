"""建立無汙染的評估集與 validation set。

做三件事，對應 EDC13 / R3C64 / R2C15 等題目：
  1. 重現當初 TPE 用掉的那 222 筆，追認為 validation set 並從評估集剔除；
  2. 剔除原文逐字存在於向量資料庫中的測試樣本（含其近似重複）；
  3. 把剩下的樣本下採樣成各類等量的 test_clean。

輸出：val222.csv / test_clean.csv / exclusion_report.json / prompts.json
"""
import argparse

import numpy as np
import pandas as pd

import common as C


def load_corpus_vectors(index_path):
    """直接從既有 FAISS index 取回語料向量，不必重新 embedding。"""
    index = C.read_faiss_index(index_path)
    vecs = index.reconstruct_n(0, index.ntotal)
    return np.asarray(vecs, dtype="float32")


def l2norm(a):
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


def max_cosine_to_corpus(query_vecs, corpus_vecs, chunk=256):
    """每個 query 對語料的最大 cosine 相似度與其命中位置。"""
    q = l2norm(query_vecs)
    c = l2norm(corpus_vecs)
    best_sim = np.zeros(len(q), dtype="float32")
    best_idx = np.zeros(len(q), dtype="int64")
    for s in range(0, len(q), chunk):
        sims = q[s:s + chunk] @ c.T
        best_idx[s:s + chunk] = sims.argmax(axis=1)
        best_sim[s:s + chunk] = sims.max(axis=1)
    return best_sim, best_idx


def per_class(df):
    return {k: int(v) for k, v in df["status"].value_counts().sort_index().items()}


def provenance_chain():
    """從原始 Kaggle CSV 重建完整的資料組成鏈（R2C13 / R2C25 / R3C24）。

    R3C24 要的是「排除 suicidal / stress / personality 之前與之後」的分佈，
    所以必須從 7 類的原始檔算起，而不是從已經切好的 test split 算起。
    數字一律現算，不寫死。
    """
    if not C.RAW_CSV.exists():
        return {"error": f"找不到原始檔 {C.RAW_CSV}"}
    raw = pd.read_csv(C.RAW_CSV)
    kept = C.VALID_LABELS
    four = raw[raw["status"].isin(kept)]
    clean = four.dropna(subset=["statement"])
    bal = pd.read_csv(C.BALANCED_CSV) if C.BALANCED_CSV.exists() else None
    tr = pd.read_csv(C.TRAIN_CSV, usecols=["statement", "status"])
    te = pd.read_csv(C.TEST_CSV, usecols=["statement", "status"])

    chain = {
        "raw_all_7_classes": {"n": len(raw), "per_class": per_class(raw)},
        "excluded_classes": {
            "classes": [c for c in raw["status"].unique() if c not in kept],
            "per_class": per_class(raw[~raw["status"].isin(kept)]),
            "reason": ("stress 非情緒障礙標籤；personality disorder 與 bipolar 特徵重疊；"
                       "suicidal 會觸發 Llama 3.1 內建安全機制而無法穩定分類"),
        },
        "kept_4_classes": {"n": len(four), "per_class": per_class(four)},
        "after_dropna_statement": {"n": len(clean), "per_class": per_class(clean),
                                   "n_dropped": len(four) - len(clean)},
        "after_balancing": ({"n": len(bal), "per_class": per_class(bal),
                             "method": "各類下採樣至最小類數量"} if bal is not None else None),
        "split_train": {"n": len(tr), "per_class": per_class(tr)},
        "split_test": {"n": len(te), "per_class": per_class(te)},
        "split_method": "stratified 80/20, random_state=42",
    }
    # 原始檔的資料品質，說明為何需要去重討論
    chain["raw_quality"] = {
        "n_null_statement": int(raw["statement"].isna().sum()),
        "n_duplicate_statement": int(raw["statement"].duplicated().sum()),
        "note": ("原始檔含重複文本且切分前未去重，是 134 筆測試樣本落入檢索語料的根因"),
    }
    return chain


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--near-dup-threshold", type=float, default=0.95,
                    help="超過此 cosine 相似度的測試樣本視為近似重複而剔除")
    ap.add_argument("--no-embed", action="store_true",
                    help="跳過 embedding 近似重複掃描（僅供本機試跑；正式執行不可使用）")
    args = ap.parse_args()
    out = C.run_dir(args.run)

    # ---------- 1. 讀檔，建立穩定的 item_id ----------
    # usecols 與原始程式一致，確保下方的 sample() 能逐筆重現
    test = pd.read_csv(C.TEST_CSV, usecols=["statement", "status"])
    train = pd.read_csv(C.TRAIN_CSV, usecols=["statement", "status"])
    test = test.reset_index(drop=True)
    test["item_id"] = test.index                       # 測試檔中的列序，全程不變
    test["key"] = test["statement"].map(C.norm_key)
    train["key"] = train["statement"].map(C.norm_key)
    C.log(f"test={len(test)}  train={len(train)}")

    # ---------- 2. 重現 TPE 當初用掉的 222 筆 ----------
    # prompt最佳化.py: pd.read_csv(..., usecols=[...]).sample(frac=0.1, random_state=42)
    val222_idx = test.sample(frac=0.1, random_state=42).index
    test["in_val222"] = test.index.isin(val222_idx)
    C.log(f"val222 = {int(test.in_val222.sum())} 筆（追認為 validation set）")

    # ---------- 3. 逐字汙染 ----------
    train_keys = set(train["key"])
    test["exact_dup"] = test["key"].isin(train_keys)
    C.log(f"逐字存在於向量資料庫中的測試樣本 = {int(test.exact_dup.sum())}")

    # ---------- 4. 近似重複（paraphrase / backtranslation 路徑，R3C64） ----------
    near_dup_report = {"method": "skipped", "threshold": args.near_dup_threshold}
    test["near_dup"] = False
    if args.no_embed:
        C.log("!! 跳過近似重複掃描（--no-embed）；正式執行必須跑這一步")
    else:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(C.EMBED_MODEL)
        # 只有仍在候選中的樣本需要掃描
        pool = test[~test.in_val222 & ~test.exact_dup]
        qv = np.asarray(embedder.encode(pool["statement"].tolist()), dtype="float32")

        per_index = {}
        worst = np.zeros(len(pool), dtype="float32")
        for name, idx_path, docs_path in (
            ("noaug", C.INDEX_NOAUG, C.DOCS_NOAUG),
            ("aug", C.INDEX_AUG, C.DOCS_AUG),
        ):
            cv = load_corpus_vectors(idx_path)
            sim, hit = max_cosine_to_corpus(qv, cv)
            worst = np.maximum(worst, sim)
            per_index[name] = {
                "corpus_size": int(len(cv)),
                "n_ge_0.90": int((sim >= 0.90).sum()),
                "n_ge_0.95": int((sim >= 0.95).sum()),
                "max_similarity": float(sim.max()),
                "median_similarity": float(np.median(sim)),
            }
            C.log(f"  [{name}] 語料 {len(cv)} 筆；≥0.90: {per_index[name]['n_ge_0.90']}，"
                  f"≥0.95: {per_index[name]['n_ge_0.95']}，max={sim.max():.4f}")

        flagged = pool.index[worst >= args.near_dup_threshold]
        test.loc[flagged, "near_dup"] = True
        near_dup_report = {
            "method": f"cosine on {C.EMBED_MODEL} embeddings vs reconstructed FAISS vectors",
            "threshold": args.near_dup_threshold,
            "n_excluded": int(len(flagged)),
            "per_index": per_index,
        }
        C.log(f"近似重複剔除 = {len(flagged)} 筆（門檻 {args.near_dup_threshold}）")

    # ---------- 5. 下採樣成各類等量 ----------
    eligible = test[~test.in_val222 & ~test.exact_dup & ~test.near_dup]
    C.log(f"乾淨候選池 = {len(eligible)}  {per_class(eligible)}")
    n_per_class = int(eligible["status"].value_counts().min())
    test_clean = (
        pd.concat([g.sample(n=n_per_class, random_state=C.SEED)
                   for _, g in eligible.groupby("status", sort=True)])
        .sort_values("item_id")
        .reset_index(drop=True)
    )
    C.log(f"test_clean = {len(test_clean)}（每類 {n_per_class}）")

    # ---------- 6. 輸出 ----------
    val222 = test[test.in_val222].sort_values("item_id")
    cols = ["item_id", "statement", "status"]
    val222[cols].to_csv(out / "val222.csv", index=False, encoding="utf-8")
    test_clean[cols].to_csv(out / "test_clean.csv", index=False, encoding="utf-8")

    # 五個條件的完整 prompt（R2C9）
    prov = C.verify_prompt_provenance()
    C.save_json(out / "prompts.json", {
        "provenance": prov,
        "llama_only": C.PROMPT_LLAMA_ONLY,
        "unoptimized": C.PROMPT_UNOPT,
        "optimized": None,  # 由 03_select_prompt.py 在 val222 上選出後填入
        "note": "optimized 欄位在 03_select_prompt.py 執行後才會有值。",
    })

    report = {
        "source_files": {
            "test_csv": str(C.TEST_CSV), "test_sha256": C.sha256_file(C.TEST_CSV),
            "train_csv": str(C.TRAIN_CSV), "train_sha256": C.sha256_file(C.TRAIN_CSV),
        },
        "language": "English",                       # R3C65
        "seed": C.SEED,
        "provenance_chain": provenance_chain(),      # R2C13 / R2C25 / R3C24
        "stages": {
            "test_original": {"n": len(test), "per_class": per_class(test)},
            "excluded_val222": {"n": int(test.in_val222.sum()),
                                "per_class": per_class(test[test.in_val222]),
                                "reason": "曾用於 TPE prompt 選擇，追認為 validation set（EDC13）"},
            "excluded_exact_dup": {"n": int(test.exact_dup.sum()),
                                   "per_class": per_class(test[test.exact_dup]),
                                   "reason": "原文逐字存在於檢索語料中，且語料條目附有 true_label（R3C64）"},
            "excluded_near_dup": {"n": int(test.near_dup.sum()),
                                  "per_class": per_class(test[test.near_dup]),
                                  "reason": "與檢索語料的 cosine 相似度超過門檻（R3C64）"},
            "eligible_pool": {"n": len(eligible), "per_class": per_class(eligible)},
            "test_clean": {"n": len(test_clean), "per_class": per_class(test_clean),
                           "n_per_class": n_per_class},
        },
        "train_per_class": per_class(train),          # R3C61
        "near_duplicate_scan": near_dup_report,
        "overlaps": {
            "test_clean_vs_val222": len(set(test_clean["item_id"]) & set(val222["item_id"])),
            "test_clean_vs_train_keys": len(set(test_clean["key"]) & train_keys),
        },
    }
    C.save_json(out / "exclusion_report.json", report)

    C.log(f"\n輸出完成 -> {out}")
    C.log(f"  重疊檢查：test_clean∩val222 = {report['overlaps']['test_clean_vs_val222']}，"
          f"test_clean∩train = {report['overlaps']['test_clean_vs_train_keys']}")


if __name__ == "__main__":
    main()
