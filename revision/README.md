# CHBR-D-26-01681 R1 — 修訂用重跑流程

回應審稿意見中「必須改程式碼才能解決」的 61 條題目。原始腳本（`../test.py`、
`../prompt最佳化.py` 等）**保持不動**作為對照，本目錄是獨立的一套流程。

鍵值採用 Reply 檔既有的流水編號：`EDC1`–`EDC23` / `R2C1`–`R2C39` / `R3C1`–`R3C69`。

## 為什麼要重跑

原始結果有四個彼此獨立的缺陷，任何一個都足以讓數字不可用：

1. **EDC13 資料洩漏**：`prompt最佳化.py` 拿 test set 的 222 筆子集當 Optuna 目標函數。
2. **逐筆預測從未存檔**：EDC17 / R2C16 要的 McNemar、bootstrap CI 在數學上無法補算。
3. **標籤解析器順序偏誤**：`for label in ['Normal',...]: if label.lower() in result.lower()`
   會把 `"This is not Normal; it is Depression"` 判成 **Normal**。
4. **Table VIII 的「最佳化 prompt」等於基準 prompt**（實測：≡ `test.py` 的 prompt
   ≡ 候選模板 #0），且真正勝出的模板只 `print` 到 console、study 未持久化而遺失。

## 核心設計：靠剔除而非重建

不重建資料集、不重建索引、不重新生成擴增資料，改為把有問題的樣本從評估集剔除：

```
舊 test                2,222
  − TPE 選擇集           222   （追認為 validation set）
  − 逐字汙染             134   （原文在向量庫中，且條目附有 true_label）
  − 近似重複               2   （cosine ≥ 0.95）
= 乾淨候選池           1,882
→ 各類下採樣           1,700   （425 × 4）
```

三個集合（train / val222 / test_clean）互斥，且 `test_clean ∩ train = 0`。

## 環境

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r revision/requirements.txt   # Windows
# .venv/bin/python -m pip install -r revision/requirements.txt         # Linux
```

**必須從 `revision/` 目錄執行**，不可從專案根目錄執行 —— 根目錄下的 `faiss/`
資料夾會把 `faiss` 套件名稱遮蔽成 namespace package（`import_faiss()` 會擋下並報錯）。

環境變數：`CHBR_ROOT`（專案根，預設為本目錄的上層）、`CHBR_RUNS`（輸出根，預設 `<root>/runs`）。

## 執行順序

```bash
RUN=R1FIX
python 00_env_probe.py     --run $RUN      # 探測 quantization/digest/硬體等（需 Ollama）
python 01_build_eval_set.py --run $RUN     # 建立 val222 / test_clean（需 sentence-transformers）
python 02_aug_audit.py     --run $RUN      # 擴增稽核 + 產生人工稽核表
python 03_select_prompt.py --run $RUN      # 在 val222 上重選 prompt   （~6,900 calls, ~2.9h）
python 04_run_conditions.py --run $RUN     # 五條件 × 1,700 + determinism（~9,100 calls, ~3.8h）
python 05_metrics.py       --run $RUN      # 雙分母指標
python 06_stats.py         --run $RUN      # McNemar + paired bootstrap + effect size
python 07_baselines.py     --run $RUN      # TF-IDF / MiniLM / DistilBERT
python 08_figures_tables.py --run $RUN     # 所有表圖（數字一律由結果檔生成）
python 09_verify_all.py    --run $RUN      # 驗收 gate，必須 0 FAIL
```

00–02 不呼叫 LLM，可在任何機器跑；03–04 需要 Ollama 與 GPU；07 的 DistilBERT
需要 GPU（可用 `--skip-distilbert` 略過）。03 與 04 都可中斷續跑。

**沒有通過 `09_verify_all.py`（0 FAIL、0 SKIP）之前，不得引用任何數字。**

## 輸出

| 檔案 | 內容 | 對應題目 |
|---|---|---|
| `run_manifest.json` | 所有 implementation settings | EDC14, R3C11–13, R3C23 |
| `exclusion_report.json` | 各階段 per-class N、剔除理由、近似重複掃描 | EDC13, R2C7/13/25, R3C24/61/64/65 |
| `aug_audit.json` | 生成樣板率、標籤詞率、過濾統計 | EDC15, R2C10/11 |
| `label_preservation_sheet.csv` | 200 筆人工稽核表（**需兩位評分者填寫後回收**） | R2C11 |
| `best_prompt.json` | 勝出模板全文、SHA256、各輪準確率 | R2C9, R3C33 |
| `preds/*.jsonl` | **逐筆預測 —— 上一輪缺的就是這個，務必永久保存** | EDC17, R2C16 |
| `metrics.json` | 兩種 invalid 分母慣例下的完整指標 | EDC18, R2C23, R3C18 |
| `stats.json` | McNemar + Holm + bootstrap CI + effect size | EDC17, R2C16/17, R3C15/21/47 |
| `baselines.json` | 三個 baseline 的 mean±sd 與訓練/推論成本 | EDC19, R3C17/34/48 |
| `tables/`, `figures/` | 論文用表圖 | R3C20/21/22/25/26/27/62/69 |
| `acceptance_report.md` | 驗收結果逐條 PASS/FAIL | — |

## 已知的誠實揭露事項

論文與回覆中必須寫明，不得省略：

- **Llama-only 條件的 prompt 是重建的**。原始程式碼未包含該條件的腳本；此處以未最佳化
  prompt 去除 reference-context 區塊重建，已在 `run_manifest.json` 標記 `reconstructed: true`。
- **檢索度量不是 cosine**。`encode()` 未做正規化，搭配 `IndexFlatL2` 實際上是
  「未正規化向量的平方 L2」。沿用既有索引，故如實記載而不更動。
- **原始「300 prompt variations」與程式碼不符**，實際候選數為 122。
- 若 `best_prompt.json` 的 `identical_to_unoptimized_prompt` 為 true，代表 C2/C4 與
  C3/C5 塌成同一條件，RQ4 與 RQ5 無法成立，必須與作者確認如何陳述。

## 相對於原始腳本修掉的缺陷

| 缺陷 | 原始 | 本流程 |
|---|---|---|
| 標籤解析順序偏誤 | 依清單順序取第一個命中 | 嚴格比對 → 唯一提及 → 否則 INVALID（附 `invalid_reason`） |
| invalid 處理不一致 | `test.py` 排除、`prompt最佳化.py` 納入 | 統一，且兩種分母慣例並陳 |
| `num_ctx` 未設定 | 使用 Ollama 預設 2048，長文件靜默截斷 | 明確指定並斷言未觸頂 |
| 搜尋結果遺失 | 只 `print`，study 未持久化 | 每次 (模板, 樣本) 評估寫入 JSONL |
| faiss 讀不了中文檔名 | — | `read_faiss_index()` 改由 Python 讀 bytes 後反序列化 |
| `faiss/` 資料夾遮蔽套件 | — | `import_faiss()` 當場擋下並說明 |
| 硬編碼 `/home/aiuser/wy/...` | 無法換機器 | 以 `CHBR_ROOT` / `CHBR_RUNS` 解析 |
