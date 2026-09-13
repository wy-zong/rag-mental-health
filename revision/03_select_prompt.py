"""在 validation set 上選出最佳化 prompt（EDC13 / R2C9 / R3C33）。

與舊版 prompt最佳化.py 的關鍵差異：
  * 舊版直接拿 test set 的子集當目標函數 —— 這正是 EDC13 指控的 test-set
    optimization。本版**只讀 val222.csv**，並斷言不碰到 test_clean。
  * 舊版只把勝出模板 `print` 到 console，study 未持久化，結果已永久遺失。
    本版把每一次 (模板, 樣本) 評估都寫進 JSONL，勝出者另存 best_prompt.json。

預設用窮舉加淘汰（exhaustive）：122 個候選本來就不多，全評一輪比 TPE 取樣
更完整也更好辯護。若要維持論文原本的 TPE 敘述，可用 --search tpe。
"""
import argparse
import time

import pandas as pd

import common as C


def stratified_subset(df, n, seed):
    """各類均分地抽出 n 筆；n 無法整除時以四捨五入後補足。"""
    per = max(1, n // df["status"].nunique())
    picked = pd.concat([g.sample(n=min(per, len(g)), random_state=seed)
                        for _, g in df.groupby("status", sort=True)])
    return picked.sort_values("item_id")


class Evaluator:
    """負責實際呼叫模型，並快取 (模板, 樣本) 與每筆樣本的檢索結果。"""

    def __init__(self, rag, manifest, model, log_writer, top_k=1):
        self.rag = rag
        self.opts = C.ollama_options(manifest)
        self.model = model
        self.log = log_writer
        self.top_k = top_k
        self.ctx_cache = {}      # item_id -> (reference_context, source_keys)
        self.cache = {}          # (tpl_idx, item_id) -> bool 是否答對
        self.n_calls = 0

    def context_for(self, item_id, statement):
        # 同一筆樣本在不同模板下的檢索結果完全相同，只需算一次
        if item_id not in self.ctx_cache:
            self.ctx_cache[item_id] = self.rag.retrieve_context(statement, k=self.top_k)
        return self.ctx_cache[item_id]

    def prime_from_log(self, path):
        """從既有 JSONL 回填快取，讓中斷的搜尋可以續跑。"""
        if not path.exists():
            return 0
        rows = C.read_jsonl(path)
        for r in rows:
            self.cache[(r["template_idx"], r["item_id"])] = r["correct"]
        return len(rows)

    def evaluate(self, tpl_idx, template, rows, round_name):
        correct = invalid = 0
        for r in rows:
            key = (tpl_idx, r["item_id"])
            if key in self.cache:
                if self.cache[key]:
                    correct += 1
                continue
            ctx, src_keys = self.context_for(r["item_id"], r["statement"])
            prompt = C.build_prompt(template, r["statement"], ctx)
            res = C.ollama_chat(prompt, self.model, self.opts)
            self.n_calls += 1
            label, reason, mode = C.parse_label(res["raw_response"])
            if res["error"]:
                label, reason = None, "api_error"
            is_correct = (label == r["status"])
            is_invalid = label is None
            correct += int(is_correct)
            invalid += int(is_invalid)
            self.cache[key] = is_correct
            self.log.write({
                "round": round_name, "template_idx": tpl_idx,
                "item_id": r["item_id"], "true_label": r["status"],
                "raw_response": res["raw_response"], "parsed_label": label,
                "invalid_reason": reason, "parse_mode": mode,
                "correct": is_correct, "is_invalid": is_invalid,
                "prompt_eval_count": res.get("prompt_eval_count"),
                "eval_count": res.get("eval_count"),
                "total_duration_ns": res.get("total_duration_ns"),
            })
        # 命中快取的樣本不重複計 invalid，故 invalid 僅供本輪參考
        return {"accuracy": correct / len(rows), "n": len(rows), "invalid": invalid}


def main():
    ap = argparse.ArgumentParser()
    C.add_run_arg(ap)
    ap.add_argument("--search", choices=["exhaustive", "tpe"], default="exhaustive")
    ap.add_argument("--round1-n", type=int, default=40, help="第一輪每個模板評估的樣本數")
    ap.add_argument("--keep", type=int, default=8, help="第一輪保留幾個模板進入第二輪")
    ap.add_argument("--trials", type=int, default=250, help="僅 --search tpe 使用")
    ap.add_argument("--model", default="llama3.1")
    ap.add_argument("--top-k", type=int, default=1)
    args = ap.parse_args()
    out = C.run_dir(args.run)

    manifest = C.load_json(out / "run_manifest.json")
    val = pd.read_csv(out / "val222.csv")
    test_ids = set(pd.read_csv(out / "test_clean.csv")["item_id"])

    # EDC13 的核心保證：搜尋過程絕不可碰到評估集
    overlap = set(val["item_id"]) & test_ids
    if overlap:
        C.die(f"validation 與 test_clean 有 {len(overlap)} 筆重疊，拒絕執行")
    C.log(f"validation={len(val)} 筆，與 test_clean 重疊 0 筆")

    templates = C.load_candidate_templates()
    C.log(f"候選模板 {len(templates)} 個；搜尋方式={args.search}")

    rag = C.RagIndex(C.INDEX_NOAUG, C.DOCS_NOAUG)
    log_path = out / "prompt_search_log.jsonl"
    resumed = 0
    if log_path.exists():
        C.log(f"偵測到既有搜尋紀錄，將續跑：{log_path}")

    rows_all = val.to_dict("records")
    rows_r1 = stratified_subset(val, args.round1_n, C.SEED).to_dict("records")
    t0 = time.time()

    # 續跑時先讀舊紀錄再以附加模式開檔
    mode_writer = C.JsonlWriter(log_path if not log_path.exists()
                                else out / "prompt_search_log.part2.jsonl")
    ev = Evaluator(rag, manifest, args.model, mode_writer, top_k=args.top_k)
    if log_path.exists():
        resumed = ev.prime_from_log(log_path)
        C.log(f"回填 {resumed} 筆既有評估結果")

    try:
        if args.search == "exhaustive":
            C.log(f"\n[第一輪] {len(templates)} 模板 × {len(rows_r1)} 樣本")
            r1 = []
            for i, tpl in enumerate(templates):
                s = ev.evaluate(i, tpl, rows_r1, "round1")
                r1.append({"template_idx": i, **s})
                if (i + 1) % 10 == 0:
                    C.log(f"  {i+1}/{len(templates)} 模板完成，累計 {ev.n_calls} 次呼叫，"
                          f"{time.time()-t0:.0f}s")
            r1.sort(key=lambda d: (-d["accuracy"], d["invalid"]))
            finalists = [d["template_idx"] for d in r1[:args.keep]]
            C.log(f"\n[第一輪] 前 {args.keep} 名模板 = {finalists}")
            C.log(f"  最佳 {r1[0]['accuracy']:.4f} / 最差 {r1[-1]['accuracy']:.4f}"
                  f"（差距 {r1[0]['accuracy']-r1[-1]['accuracy']:.4f}）")

            C.log(f"\n[第二輪] {len(finalists)} 模板 × {len(rows_all)} 樣本")
            r2 = []
            for i in finalists:
                s = ev.evaluate(i, templates[i], rows_all, "round2")
                r2.append({"template_idx": i, **s})
                C.log(f"  模板 {i}: acc={s['accuracy']:.4f}")
            r2.sort(key=lambda d: (-d["accuracy"], d["invalid"]))
            best_idx = r2[0]["template_idx"]
            rounds = {"round1": r1, "round2": r2}
            study_path = None
        else:
            import optuna
            study_path = out / "prompt_study.db"
            study = optuna.create_study(
                direction="maximize",
                sampler=optuna.samplers.TPESampler(seed=C.SEED),
                storage=f"sqlite:///{study_path}",     # 舊版沒有這行，結果才會遺失
                study_name=f"prompt_{args.run}", load_if_exists=True)

            def objective(trial):
                idx = trial.suggest_categorical("template_idx", list(range(len(templates))))
                return ev.evaluate(idx, templates[idx], rows_r1, "tpe")["accuracy"]

            study.optimize(objective, n_trials=args.trials)
            best_idx = study.best_trial.params["template_idx"]
            rounds = {"tpe_best_value": study.best_trial.value,
                      "n_trials": len(study.trials)}
            r2 = [{"template_idx": best_idx, "accuracy": study.best_trial.value,
                   "n": len(rows_r1), "invalid": None}]
    finally:
        mode_writer.close()

    best_template = templates[best_idx]
    identical_to_unopt = (
        C._norm_prompt(best_template) == C._norm_prompt(C.PROMPT_UNOPT))

    C.save_json(out / "best_prompt.json", {
        "search": args.search,
        "selection_data": "val222.csv（那 222 筆曾用於舊版 TPE，已追認為 validation set）",
        "selection_overlap_with_test": 0,
        "n_candidate_templates": len(templates),
        "best_template_idx": best_idx,
        "best_template_sha256": C.sha256_text(best_template),
        "best_template_text": best_template,
        "best_val_accuracy": r2[0]["accuracy"],
        "identical_to_unoptimized_prompt": identical_to_unopt,
        "identical_note": (
            "若為 true，代表最佳化 prompt 與未最佳化 prompt 相同，"
            "C2/C4 與 C3/C5 會塌成同一條件，必須在論文中據實說明。"),
        "rounds": rounds,
        "n_llm_calls": ev.n_calls,
        "resumed_evaluations": resumed,
        "elapsed_sec": round(time.time() - t0, 1),
        "study_db": str(study_path) if study_path else None,
    })

    prompts = C.load_json(out / "prompts.json")
    prompts["optimized"] = best_template
    prompts["optimized_template_idx"] = best_idx
    prompts["note"] = "optimized 由 03_select_prompt.py 在 val222 上選出。"
    C.save_json(out / "prompts.json", prompts)

    C.log(f"\n勝出模板 #{best_idx}，val accuracy={r2[0]['accuracy']:.4f}")
    C.log(f"與未最佳化 prompt 相同？ {identical_to_unopt}")
    C.log(f"總呼叫 {ev.n_calls} 次，耗時 {(time.time()-t0)/60:.1f} 分鐘")
    if identical_to_unopt:
        C.log("!! 勝出模板等於基準 prompt —— 五條件會塌成三條件，需與作者確認如何陳述")


if __name__ == "__main__":
    main()
