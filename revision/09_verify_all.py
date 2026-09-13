"""單一驗收 gate：跑完計畫 §4 的每一條斷言，任一失敗即以非零結束。

輸出 acceptance_report.md，每條一行 PASS/FAIL。這是「可以把數字交給論文端」
的唯一判準 —— 沒有通過就不得引用任何數字。
"""
import argparse

import numpy as np
import pandas as pd

import common as C


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, step, name, ok, detail=""):
        self.rows.append({"step": step, "name": name,
                          "ok": bool(ok), "detail": str(detail)})
        C.log(f"  [{'PASS' if ok else 'FAIL'}] {step} {name}"
              + (f" — {detail}" if detail else ""))
        return ok

    def skip(self, step, name, why):
        self.rows.append({"step": step, "name": name, "ok": None, "detail": why})
        C.log(f"  [SKIP] {step} {name} — {why}")

    def waive(self, step, name, why):
        """跟 skip 不同：這不是「還沒做完」，是**刻意決定不做**且已在回覆信中
        說明。不計入 failed/skipped，不會擋住驗收 gate，但仍完整記錄在
        acceptance_report.md 裡，避免變成靜默消失、日後被誤認為疏漏。"""
        self.rows.append({"step": step, "name": name, "ok": "waived", "detail": why})
        C.log(f"  [WAIVED] {step} {name} — {why}")

    @property
    def failed(self):
        return [r for r in self.rows if r["ok"] is False]

    @property
    def skipped(self):
        return [r for r in self.rows if r["ok"] is None]

    @property
    def waived(self):
        return [r for r in self.rows if r["ok"] == "waived"]


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--allow-skips", action="store_true",
                    help="允許尚未執行的步驟以 SKIP 計（開發中使用）")
    args = ap.parse_args()
    out = C.run_dir(args.run)
    k = Checks()
    exists = lambda p: (out / p).exists()

    # ---------------- 00 run_manifest ----------------
    C.log("\n== 00 env probe ==")
    if not exists("run_manifest.json"):
        k.skip("00", "run_manifest.json 存在", "尚未執行 00_env_probe.py")
        manifest = None
    else:
        manifest = C.load_json(out / "run_manifest.json")
        llm = manifest.get("llm", {})
        k.check("00", "quantization 已實際查出", bool(llm.get("quantization")),
                llm.get("quantization"))
        k.check("00", "model digest 已實際查出", bool(llm.get("model_digest")),
                str(llm.get("model_digest"))[:20])
        k.check("00", "num_ctx 已明確指定",
                bool(manifest["generation"].get("num_ctx")),
                manifest["generation"].get("num_ctx"))
        k.check("00", "git_commit 已記錄", bool(manifest.get("git_commit")))
        k.check("00", "pip_freeze 已記錄",
                bool(manifest.get("packages", {}).get("pip_freeze")))
        pr = manifest.get("prompts", {})
        k.check("00", "未最佳化 prompt 與 test.py 相符",
                pr.get("unoptimized_matches_testpy") is True)
        k.check("00", "候選模板數 = 122", pr.get("n_candidate_templates") == 122,
                pr.get("n_candidate_templates"))

    # ---------------- 01 評估集 ----------------
    C.log("\n== 01 build eval set ==")
    if not (exists("val222.csv") and exists("test_clean.csv")):
        k.skip("01", "評估集檔案存在", "尚未執行 01_build_eval_set.py")
        val = test = None
    else:
        val = pd.read_csv(out / "val222.csv")
        test = pd.read_csv(out / "test_clean.csv")
        rep = C.load_json(out / "exclusion_report.json")
        train = pd.read_csv(C.TRAIN_CSV, usecols=["statement", "status"])
        train_keys = set(train["statement"].map(C.norm_key))

        k.check("01", "val222 恰 222 筆", len(val) == 222, len(val))
        # 與原始 sample(frac=0.1, random_state=42) 逐筆相符
        raw = pd.read_csv(C.TEST_CSV, usecols=["statement", "status"]).reset_index(drop=True)
        repro = set(raw.sample(frac=0.1, random_state=42).index)
        k.check("01", "val222 與原始 TPE 抽樣完全相同",
                set(val["item_id"]) == repro)

        counts = test["status"].value_counts()
        k.check("01", "test_clean 各類等量", counts.nunique() == 1,
                {str(lab): int(v) for lab, v in counts.items()})
        k.check("01", "test_clean ∩ val222 = 0",
                len(set(test["item_id"]) & set(val["item_id"])) == 0)
        k.check("01", "test_clean ∩ train（逐字）= 0",
                len(set(test["statement"].map(C.norm_key)) & train_keys) == 0)
        k.check("01", "exclusion_report 記錄 train per-class support（R3C61）",
                bool(rep.get("train_per_class")))
        k.check("01", "exclusion_report 記錄資料語言（R3C65）",
                rep.get("language") == "English")
        nd = rep.get("near_duplicate_scan", {})
        k.check("01", "近似重複掃描已實際執行（R3C64）",
                nd.get("method") != "skipped", nd.get("method"))

    # ---------------- 02 擴增稽核 ----------------
    C.log("\n== 02 augmentation audit ==")
    if not exists("aug_audit.json"):
        k.skip("02", "aug_audit.json 存在", "尚未執行 02_aug_audit.py")
    else:
        a = C.load_json(out / "aug_audit.json")
        k.check("02", "改寫文本 17772 筆", a.get("n_paraphrases") == 17772,
                a.get("n_paraphrases"))
        k.check("02", "三元組結構已驗證", a.get("triplet_structure_verified") is True)
        k.check("02", "過濾後保留 ≥ 50%",
                a["filters"]["retained"] >= 0.5 * a["n_paraphrases"],
                f"{a['filters']['retained']}/{a['n_paraphrases']}")
        lp = out / "label_preservation.json"
        if lp.exists():
            d = C.load_json(lp)
            k.check("02", "人工稽核 ≥200 筆且雙評分者",
                    d.get("n_rated", 0) >= 200 and d.get("n_raters") == 2)
            k.check("02", "已回報 Cohen's kappa", d.get("kappa") is not None)
        else:
            # 2026-09-14 決定：EDC15 / R2C11 要求的人工 label-preservation
            # 稽核因內部死線時間不足，主動決定不做，已在回覆信中如實說明並
            # 承認為本次修訂的限制（非疏漏）。故意 waive 而非 skip：
            # 不擋驗收 gate（03-08 的其餘數字與此無關），但仍完整留下紀錄。
            k.waive("02", "label_preservation.json（人工稽核）",
                    "EDC15/R2C11：時間不足，主動決定不做，已在回覆信中揭露為限制")

    # ---------------- 03 prompt 選擇 ----------------
    C.log("\n== 03 select prompt ==")
    if not exists("best_prompt.json"):
        k.skip("03", "best_prompt.json 存在", "尚未執行 03_select_prompt.py")
    else:
        bp = C.load_json(out / "best_prompt.json")
        log_rows = []
        for f in ("prompt_search_log.jsonl", "prompt_search_log.part2.jsonl"):
            if exists(f):
                log_rows += C.read_jsonl(out / f)
        evaluated = {r["item_id"] for r in log_rows}
        k.check("03", "搜尋只用到 val222",
                bool(val is not None) and evaluated <= set(val["item_id"]),
                f"評估過 {len(evaluated)} 個 item")
        k.check("03", "搜尋樣本 ∩ test_clean = 0",
                bool(test is not None) and not (evaluated & set(test["item_id"])))
        k.check("03", "勝出模板已記錄全文與 SHA256",
                bool(bp.get("best_template_text")) and bool(bp.get("best_template_sha256")))
        if bp.get("identical_to_unoptimized_prompt"):
            k.check("03", "勝出模板與基準 prompt 不同", False,
                    "勝出模板等於未最佳化 prompt，C2/C4 與 C3/C5 會塌成同一條件")
        else:
            k.check("03", "勝出模板與基準 prompt 不同", True)

    # ---------------- 04 條件執行 ----------------
    C.log("\n== 04 run conditions ==")
    preds = sorted((out / "preds").glob("*.jsonl")) if (out / "preds").exists() else []
    frames = {}
    if not preds:
        k.skip("04", "preds/*.jsonl 存在", "尚未執行 04_run_conditions.py")
    else:
        for p in preds:
            frames[p.stem] = pd.DataFrame(C.read_jsonl(p))
        k.check("04", "五個條件都有預測檔", len(frames) == 5, list(frames))
        n_expected = len(test) if test is not None else None
        k.check("04", "每個條件的筆數 = test_clean",
                all(len(v) == n_expected for v in frames.values()),
                {k2: len(v) for k2, v in frames.items()})
        id_sets = [set(v["item_id"]) for v in frames.values()]
        k.check("04", "五條件 item_id 集合完全相同（paired test 前提）",
                all(s == id_sets[0] for s in id_sets))
        k.check("04", "100% 記錄 raw_response（或至少記錄了錯誤原因）",
                all((v["raw_response"].notna() | v["error"].notna()).all()
                    for v in frames.values()))
        k.check("04", "無任何檢索洩漏到評估集（R3C64）",
                all(not v["retrieval_leak"].any() for v in frames.values()))
        if manifest:
            ctx = manifest["generation"]["num_ctx"]
            worst = max((v["prompt_eval_count"].max() for v in frames.values()
                         if v["prompt_eval_count"].notna().any()), default=0)
            k.check("04", "prompt token 未觸及 num_ctx（無靜默截斷）",
                    worst < ctx, f"max={worst} / num_ctx={ctx}")
        if exists("conditions_summary.json"):
            det = C.load_json(out / "conditions_summary.json").get("determinism", {})
            k.check("04", "determinism 一致率已量測（R3C14）",
                    det.get("exact_agreement_rate") is not None,
                    det.get("exact_agreement_rate"))

    # ---------------- 05 指標 ----------------
    C.log("\n== 05 metrics ==")
    if not exists("metrics.json"):
        k.skip("05", "metrics.json 存在", "尚未執行 05_metrics.py")
    else:
        m = C.load_json(out / "metrics.json")
        k.check("05", "invalid 定義已寫入（EDC18）", bool(m.get("invalid_definition")))
        ok_denom = all(
            r["convention_A_primary"]["denominator"] == r["n"]
            for r in m["results"].values())
        k.check("05", "慣例 A 分母 = 全部樣本數", ok_denom)
        ok_support = all(
            sum(c["support"] for c in r["convention_A_primary"]["per_class"].values()) == r["n"]
            for r in m["results"].values())
        k.check("05", "per-class support 加總 = 樣本數", ok_support)
        k.check("05", "兩種分母慣例都已提供（R2C23）",
                all("convention_B_secondary" in r for r in m["results"].values()))

    # ---------------- 06 統計 ----------------
    C.log("\n== 06 stats ==")
    if not exists("stats.json"):
        k.skip("06", "stats.json 存在", "尚未執行 06_stats.py")
    else:
        s = C.load_json(out / "stats.json")
        k.check("06", "已回報 McNemar 與 Holm 校正 p 值",
                all("holm_adjusted_p" in c["mcnemar"] for c in s["comparisons"]))
        k.check("06", "每個比較都有 bootstrap 95% CI",
                all(c.get("delta_accuracy_ci95") for c in s["comparisons"]))
        k.check("06", "已回報效果量（相異樣本數，R2C17）",
                all("n_samples_with_different_outcome" in c for c in s["comparisons"]))
        # b+c 必須與直接從 JSONL 重算的相異數一致
        if frames:
            ok = True
            for c in s["comparisons"]:
                lo, hi = frames.get(c["from"]), frames.get(c["to"])
                if lo is None or hi is None:
                    continue
                a_ok = (lo["parsed_label"] == lo["true_label"]).to_numpy()
                b_ok = (hi["parsed_label"] == hi["true_label"]).to_numpy()
                recomputed = int(np.count_nonzero(a_ok ^ b_ok))
                ok &= (recomputed == c["mcnemar"]["n_discordant"])
            k.check("06", "b+c 與由 JSONL 重算的相異數相符", ok)

    # ---------------- 07 baselines ----------------
    C.log("\n== 07 baselines ==")
    if not exists("baselines.json"):
        k.skip("07", "baselines.json 存在", "尚未執行 07_baselines.py")
    else:
        b = C.load_json(out / "baselines.json")
        k.check("07", "已提供 ≥2 個 baseline（EDC19）", len(b.get("baselines", {})) >= 2,
                list(b.get("baselines", {})))
        k.check("07", "已記錄訓練/推論時間（cost 比較）",
                all("train_sec" in v for v in b.get("baselines", {}).values()))

    # ---------------- 輸出 ----------------
    lines = ["# Acceptance report", "",
             f"- run: `{args.run}`",
             f"- PASS {sum(1 for r in k.rows if r['ok'] is True)}"
             f" / FAIL {len(k.failed)} / SKIP {len(k.skipped)}"
             f" / WAIVED {len(k.waived)}", "",
             "| step | check | result | detail |", "|---|---|---|---|"]
    for r in k.rows:
        mark = {True: "PASS", False: "**FAIL**", None: "skip",
                "waived": "WAIVED"}[r["ok"]]
        lines.append(f"| {r['step']} | {r['name']} | {mark} | {r['detail'][:80]} |")
    (out / "acceptance_report.md").write_text("\n".join(lines), encoding="utf-8")

    C.log(f"\n{'='*60}")
    C.log(f"PASS {sum(1 for r in k.rows if r['ok'] is True)}  "
          f"FAIL {len(k.failed)}  SKIP {len(k.skipped)}  WAIVED {len(k.waived)}")
    C.log(f"報告 -> {out / 'acceptance_report.md'}")
    if k.failed:
        C.die(f"{len(k.failed)} 項驗收失敗，不得引用任何數字")
    if k.skipped and not args.allow_skips:
        C.die(f"{len(k.skipped)} 項尚未執行；全部完成後才算通過"
              "（開發中可加 --allow-skips）")
    if k.waived:
        C.log(f"!! {len(k.waived)} 項刻意 waive（非疏漏，已於回覆信揭露），"
              "詳見 acceptance_report.md，回覆信與論文措辭需與此一致")
    C.log("全部通過。")


if __name__ == "__main__":
    main()
