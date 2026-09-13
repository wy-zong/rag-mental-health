# Acceptance report

- run: `FULL`
- PASS 13 / FAIL 0 / SKIP 6

| step | check | result | detail |
|---|---|---|---|
| 00 | run_manifest.json 存在 | skip | 尚未執行 00_env_probe.py |
| 01 | val222 恰 222 筆 | PASS | 222 |
| 01 | val222 與原始 TPE 抽樣完全相同 | PASS |  |
| 01 | test_clean 各類等量 | PASS | {'Anxiety': 425, 'Depression': 425, 'Bipolar': 425, 'Normal': 425} |
| 01 | test_clean ∩ val222 = 0 | PASS |  |
| 01 | test_clean ∩ train（逐字）= 0 | PASS |  |
| 01 | exclusion_report 記錄 train per-class support（R3C61） | PASS |  |
| 01 | exclusion_report 記錄資料語言（R3C65） | PASS |  |
| 01 | 近似重複掃描已實際執行（R3C64） | PASS | cosine on all-MiniLM-L6-v2 embeddings vs reconstructed FAISS vectors |
| 02 | 改寫文本 17772 筆 | PASS | 17772 |
| 02 | 三元組結構已驗證 | PASS |  |
| 02 | 過濾後保留 ≥ 50% | PASS | 10444/17772 |
| 02 | label_preservation.json | skip | 人工稽核尚未回填 |
| 03 | best_prompt.json 存在 | skip | 尚未執行 03_select_prompt.py |
| 04 | preds/*.jsonl 存在 | skip | 尚未執行 04_run_conditions.py |
| 05 | metrics.json 存在 | skip | 尚未執行 05_metrics.py |
| 06 | stats.json 存在 | skip | 尚未執行 06_stats.py |
| 07 | 已提供 ≥2 個 baseline（EDC19） | PASS | ['tfidf_lr', 'minilm_lr'] |
| 07 | 已記錄訓練/推論時間（cost 比較） | PASS |  |