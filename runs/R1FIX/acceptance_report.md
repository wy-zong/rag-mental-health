# Acceptance report

- run: `R1FIX`
- PASS 39 / FAIL 0 / SKIP 0 / WAIVED 1

| step | check | result | detail |
|---|---|---|---|
| 00 | quantization 已實際查出 | PASS | Q4_K_M |
| 00 | model digest 已實際查出 | PASS | 667b0c1932bc6ffc593e |
| 00 | num_ctx 已明確指定 | PASS | 8192 |
| 00 | git_commit 已記錄 | PASS |  |
| 00 | pip_freeze 已記錄 | PASS |  |
| 00 | 未最佳化 prompt 與 test.py 相符 | PASS |  |
| 00 | 候選模板數 = 122 | PASS | 122 |
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
| 02 | label_preservation.json（人工稽核） | WAIVED | EDC15/R2C11：時間不足，主動決定不做，已在回覆信中揭露為限制 |
| 03 | 搜尋只用到 val222 | PASS | 評估過 222 個 item |
| 03 | 搜尋樣本 ∩ test_clean = 0 | PASS |  |
| 03 | 勝出模板已記錄全文與 SHA256 | PASS |  |
| 03 | 勝出模板與基準 prompt 不同 | PASS |  |
| 04 | 五個條件都有預測檔 | PASS | ['C1_llama_only', 'C2_rag_unopt_noaug', 'C3_rag_unopt_aug', 'C4_rag_opt_noaug',  |
| 04 | 每個條件的筆數 = test_clean | PASS | {'C1_llama_only': 1700, 'C2_rag_unopt_noaug': 1700, 'C3_rag_unopt_aug': 1700, 'C |
| 04 | 五條件 item_id 集合完全相同（paired test 前提） | PASS |  |
| 04 | 100% 記錄 raw_response（或至少記錄了錯誤原因） | PASS |  |
| 04 | 無任何檢索洩漏到評估集（R3C64） | PASS |  |
| 04 | prompt token 未觸及 num_ctx（無靜默截斷） | PASS | max=5017 / num_ctx=8192 |
| 04 | determinism 一致率已量測（R3C14） | PASS | 1.0 |
| 05 | invalid 定義已寫入（EDC18） | PASS |  |
| 05 | 慣例 A 分母 = 全部樣本數 | PASS |  |
| 05 | per-class support 加總 = 樣本數 | PASS |  |
| 05 | 兩種分母慣例都已提供（R2C23） | PASS |  |
| 06 | 已回報 McNemar 與 Holm 校正 p 值 | PASS |  |
| 06 | 每個比較都有 bootstrap 95% CI | PASS |  |
| 06 | 已回報效果量（相異樣本數，R2C17） | PASS |  |
| 06 | b+c 與由 JSONL 重算的相異數相符 | PASS |  |
| 07 | 已提供 ≥2 個 baseline（EDC19） | PASS | ['tfidf_lr', 'minilm_lr', 'distilbert'] |
| 07 | 已記錄訓練/推論時間（cost 比較） | PASS |  |