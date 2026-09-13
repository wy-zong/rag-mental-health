"""回讀人工稽核表，計算標籤保真率與評分者一致性（R2C11 / EDC15）。

02_aug_audit.py 產生 label_preservation_sheet.csv 之後，由兩位評分者各自
在 rater1_label_preserved / rater2_label_preserved 欄填 1（標籤有保留）或
0（未保留）。本腳本把填好的表讀回來，輸出 label_preservation.json。

編輯明確要求的是「label-preservation/error analysis」，所以這裡同時給：
  * 每位評分者與共識的保真率，附 Wilson 95% CI（小樣本比例不可用常態近似）
  * Cohen's kappa（評分者一致性）
  * 不一致與判定為「未保留」的案例，供論文舉例
"""
import argparse

import numpy as np
import pandas as pd

import common as C

R1, R2 = "rater1_label_preserved", "rater2_label_preserved"


def wilson_ci(k, n, z=1.96):
    """Wilson score 區間：n 小或比例接近 0/1 時比常態近似可靠。"""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (round(float(max(0.0, centre - half)), 4),
            round(float(min(1.0, centre + half)), 4))


def rate_block(mask_series, label=""):
    k = int(mask_series.sum())
    n = int(mask_series.notna().sum())
    lo, hi = wilson_ci(k, n)
    return {"n_rated": n, "n_preserved": k,
            "rate": round(k / n, 4) if n else None,
            "wilson_ci95": [lo, hi]}


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--sheet", default="label_preservation_sheet.csv",
                    help="填好的稽核表檔名（相對於 run 目錄）")
    args = ap.parse_args()
    out = C.run_dir(args.run)

    path = out / args.sheet
    if not path.exists():
        C.die(f"找不到 {path} —— 請先執行 02_aug_audit.py 產生稽核表")
    df = pd.read_csv(path, encoding="utf-8-sig")

    for col in (R1, R2):
        if col not in df.columns:
            C.die(f"稽核表缺少欄位 {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    both = df[df[R1].notna() & df[R2].notna()].copy()
    C.log(f"稽核表 {len(df)} 列；兩位評分者都已填的有 {len(both)} 列")
    if both.empty:
        C.die("尚無任何一列被兩位評分者同時評分 —— 人工稽核尚未完成，"
              "此步驟必須有實際評分結果，不得以預設值產生")

    invalid = both[~both[R1].isin([0, 1]) | ~both[R2].isin([0, 1])]
    if len(invalid):
        C.die(f"有 {len(invalid)} 列的評分不是 0 或 1，請修正後重跑")

    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score(both[R1].astype(int), both[R2].astype(int)))
    agreement = float((both[R1] == both[R2]).mean())

    # 共識：兩位都判定保留才算保留（保守作法，寧可低估保真率）
    both["consensus"] = ((both[R1] == 1) & (both[R2] == 1)).astype(int)

    per_label = {}
    for lab, g in both.groupby("label", sort=True):
        per_label[str(lab)] = rate_block(g["consensus"])

    disagreements = both[both[R1] != both[R2]]
    not_preserved = both[both["consensus"] == 0]

    result = {
        "sheet": str(path),
        "n_rows_in_sheet": len(df),
        "n_rated": len(both),
        "n_raters": 2,
        "kappa": round(kappa, 4),
        "kappa_interpretation": (
            "<0 poor, 0–0.20 slight, 0.21–0.40 fair, 0.41–0.60 moderate, "
            "0.61–0.80 substantial, >0.80 almost perfect (Landis & Koch)"),
        "raw_agreement": round(agreement, 4),
        "consensus_definition": "兩位評分者都判定保留才計為保留（保守）",
        "overall": rate_block(both["consensus"]),
        "per_label": per_label,
        "rater1": rate_block(both[R1]),
        "rater2": rate_block(both[R2]),
        "n_disagreements": len(disagreements),
        "examples_not_preserved": not_preserved[
            ["doc_pos", "label", "source", "paraphrase"]
        ].head(10).to_dict("records"),
        "examples_disagreed": disagreements[
            ["doc_pos", "label", "source", "paraphrase", R1, R2]
        ].head(10).to_dict("records"),
        "note": ("本結果只涵蓋抽樣的改寫文本，並非全體 17,772 筆的普查；"
                 "論文需以此措辭陳述。"),
    }
    C.save_json(out / "label_preservation.json", result)

    o = result["overall"]
    C.log(f"  Cohen's kappa = {kappa:.4f}（原始一致率 {agreement:.4f}）")
    C.log(f"  共識保真率 = {o['rate']} "
          f"（{o['n_preserved']}/{o['n_rated']}，95% CI {o['wilson_ci95']}）")
    for lab, d in per_label.items():
        C.log(f"    {lab:12} {d['rate']}  CI {d['wilson_ci95']}  (n={d['n_rated']})")
    C.log(f"\n輸出完成 -> {out / 'label_preservation.json'}")


if __name__ == "__main__":
    main()
