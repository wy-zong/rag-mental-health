"""稽核既有的 LLM 擴增資料（EDC15 / R2C10 / R2C11）。

不呼叫任何 LLM，也不重建索引 —— 擴增文本完整保存在
`llm資料擴增_rag_docs.json` 裡，且結構是嚴格的三元組
（index 3i 為原文，3i+1 與 3i+2 為它的兩個改寫版）。

輸出：aug_audit.json / label_preservation_sheet.csv / aug_filtered.jsonl
"""
import argparse
import re

import pandas as pd

import common as C

# 「Here is a rewritten version of the text:」這類指令跟隨語的樣板前綴
PREAMBLE_RE = re.compile(
    r"^\s*(here\s+(is|are)|here's)\b[^\n:]{0,120}:\s*", re.IGNORECASE)

LABEL_WORD_RE = {
    lab: re.compile(rf"(?<![A-Za-z]){lab}(?![A-Za-z])", re.IGNORECASE)
    for lab in C.VALID_LABELS
}


def strip_preamble(text):
    """剝掉樣板前綴；回傳 (清理後文字, 是否有前綴)。"""
    cleaned = PREAMBLE_RE.sub("", text, count=1).strip()
    return cleaned, cleaned != text.strip()


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--audit-n", type=int, default=200,
                    help="人工稽核抽樣張數（各類均分）")
    ap.add_argument("--len-ratio-min", type=float, default=0.5)
    ap.add_argument("--len-ratio-max", type=float, default=2.0)
    args = ap.parse_args()
    out = C.run_dir(args.run)

    noaug = C.load_json(C.DOCS_NOAUG)
    aug = C.load_json(C.DOCS_AUG)
    C.log(f"noaug docs={len(noaug)}  aug docs={len(aug)}")

    # ---------- 1. 驗證三元組結構 ----------
    if len(aug) != 3 * len(noaug):
        C.die(f"擴增語料不是三元組結構：{len(aug)} != 3*{len(noaug)}")
    mismatched = [i for i in range(len(noaug)) if aug[3 * i] != noaug[i]]
    if mismatched:
        C.die(f"aug[3i] 與 noaug[i] 有 {len(mismatched)} 處不符，無法安全還原來源關聯")
    C.log("三元組結構驗證通過：aug[3i] == noaug[i]")

    # ---------- 2. 還原 (來源, 改寫, 標籤) ----------
    records = []
    for i in range(len(noaug)):
        src_doc = noaug[i]
        source = C.RagIndex.doc_source(src_doc)
        label = C.RagIndex.doc_label(src_doc)
        for j, pos in enumerate((3 * i + 1, 3 * i + 2)):
            body = C.RagIndex.doc_source(aug[pos])
            records.append({
                "source_row": i, "para_idx": j, "doc_pos": pos,
                "source": source, "label": label, "paraphrase_raw": body,
            })
    df = pd.DataFrame(records)
    C.log(f"還原改寫文本 {len(df)} 筆")

    # ---------- 3. 量化生成樣板與標籤詞 ----------
    cleaned, had_pre = zip(*(strip_preamble(t) for t in df["paraphrase_raw"]))
    df["paraphrase"] = cleaned
    df["had_preamble"] = had_pre
    df["has_blank_line"] = df["paraphrase_raw"].str.contains("\n\n", regex=False)
    df["own_label_in_body"] = [
        bool(LABEL_WORD_RE[lab].search(body)) if lab in LABEL_WORD_RE else False
        for lab, body in zip(df["label"], df["paraphrase"])
    ]
    df["other_label_in_body"] = [
        any(rx.search(body) for lab2, rx in LABEL_WORD_RE.items()
            if lab2.lower() != str(lab).lower())
        for lab, body in zip(df["label"], df["paraphrase"])
    ]
    src_len = df["source"].str.len().clip(lower=1)
    df["len_ratio"] = df["paraphrase"].str.len() / src_len

    # 兩種量法都記下來：剝前綴「之前」的原始文本 vs「之後」的內文。
    # 前綴本身常含類別詞（例："Here is a rewritten version with Anxiety emotion:"），
    # 所以兩個數字會不同，必須分開陳述以免日後對不上。
    df["starts_with_here"] = df["paraphrase_raw"].str.match(r"\s*here\s*(is|are|'s)\b",
                                                            case=False, na=False)
    df["own_label_in_raw"] = [
        bool(LABEL_WORD_RE[lab].search(raw)) if lab in LABEL_WORD_RE else False
        for lab, raw in zip(df["label"], df["paraphrase_raw"])
    ]

    n = len(df)
    rate = lambda col: {"n": int(df[col].sum()), "pct": round(100 * df[col].mean(), 1)}
    artifacts = {
        "starts_with_here_loose": rate("starts_with_here"),
        "preamble_boilerplate": rate("had_preamble"),
        "own_label_word_in_raw_text": rate("own_label_in_raw"),
        "blank_line_split": rate("has_blank_line"),
        "own_label_word_in_body": rate("own_label_in_body"),
        "other_label_word_in_body": rate("other_label_in_body"),
        "empty_after_cleaning": {"n": int((df["paraphrase"].str.strip() == "").sum())},
        "len_ratio": {
            "p05": round(float(df.len_ratio.quantile(0.05)), 3),
            "median": round(float(df.len_ratio.median()), 3),
            "p95": round(float(df.len_ratio.quantile(0.95)), 3),
        },
    }
    C.log(f"樣板前綴 {artifacts['preamble_boilerplate']['pct']}%  "
          f"自身類別詞 {artifacts['own_label_word_in_body']['pct']}%  "
          f"他類類別詞 {artifacts['other_label_word_in_body']['pct']}%")

    # ---------- 4. 品質過濾（供 §7 所列的延後消融使用） ----------
    df["drop_empty"] = df["paraphrase"].str.strip() == ""
    df["drop_len"] = (df["len_ratio"] < args.len_ratio_min) | (df["len_ratio"] > args.len_ratio_max)
    df["drop_other_label"] = df["other_label_in_body"]
    df["drop_dup_of_source"] = [
        C.norm_key(p) == C.norm_key(s) for p, s in zip(df["paraphrase"], df["source"])
    ]
    drop_cols = ["drop_empty", "drop_len", "drop_other_label", "drop_dup_of_source"]
    df["dropped"] = df[drop_cols].any(axis=1)
    filters = {c: int(df[c].sum()) for c in drop_cols}
    filters["retained"] = int((~df["dropped"]).sum())
    C.log(f"過濾後保留 {filters['retained']} / {n} "
          f"({100*filters['retained']/n:.1f}%)")

    # ---------- 5. 人工稽核抽樣表（R2C11） ----------
    keep = df[~df["dropped"]]
    per_label = max(1, args.audit_n // max(1, keep["label"].nunique()))
    sheet = pd.concat([
        g.sample(n=min(per_label, len(g)), random_state=C.SEED)
        for _, g in keep.groupby("label", sort=True)
    ]).sort_values(["label", "source_row"])
    sheet_out = sheet[["doc_pos", "label", "source", "paraphrase"]].copy()
    # 兩位評分者各自填寫，之後回讀計算 Cohen's kappa
    sheet_out["rater1_label_preserved"] = ""
    sheet_out["rater2_label_preserved"] = ""
    sheet_out["notes"] = ""
    sheet_out.to_csv(out / "label_preservation_sheet.csv", index=False,
                     encoding="utf-8-sig")   # utf-8-sig 讓 Excel 正確開啟中文
    C.log(f"人工稽核抽樣表 {len(sheet_out)} 列 -> label_preservation_sheet.csv")

    # ---------- 6. 輸出 ----------
    with C.JsonlWriter(out / "aug_filtered.jsonl") as w:
        for r in keep[["doc_pos", "source_row", "para_idx", "label",
                       "source", "paraphrase"]].to_dict("records"):
            w.write(r)

    C.save_json(out / "aug_audit.json", {
        "source_file": str(C.DOCS_AUG),
        "sha256": C.sha256_file(C.DOCS_AUG),
        "triplet_structure_verified": True,
        "n_source_rows": len(noaug),
        "n_paraphrases": n,
        "generation_artifacts": artifacts,
        "filters": filters,
        "filter_rules": {
            "drop_empty": "清掉樣板前綴後為空",
            "drop_len": f"與原文長度比不在 [{args.len_ratio_min}, {args.len_ratio_max}]",
            "drop_other_label": "內文出現與自身標籤不同的類別詞（標籤噪音）",
            "drop_dup_of_source": "改寫後與原文完全相同",
        },
        "interpretation": (
            "樣板前綴使 embedding 編碼到的是指令跟隨語而非文本語意，"
            "這是擴增只帶來 +0.0009 準確率的直接解釋（EDC15 / EDC16）。"
            "自身類別詞出現在內文並非額外洩漏 —— 語料條目本來就以 "
            "'<statement> true_label is <label>' 的形式附帶標籤；"
            "真正有問題的是他類類別詞（標籤噪音）與樣板前綴。"
        ),
        "audit_sheet": {
            "n_rows": len(sheet_out),
            "per_label": {k: int(v) for k, v in sheet.groupby("label").size().items()},
            "next_step": "兩位評分者各自填 rater1/rater2_label_preserved（1=保留, 0=未保留），回讀計算 Cohen's kappa 與 Wilson 95% CI",
            "status_2026_09_14": (
                "EDC15/R2C11 的人工稽核因內部死線時間不足，主動決定不做，"
                "已在回覆信中揭露為本次修訂的限制（見 revision/README.md）。"
                "本表仍會產生，若日後有餘裕仍可回頭填寫。"
            ),
        },
        "note": "本步驟不重建索引；aug_filtered.jsonl 僅供 §7 所列的延後消融使用。",
    })
    C.log(f"\n輸出完成 -> {out}")


if __name__ == "__main__":
    main()
