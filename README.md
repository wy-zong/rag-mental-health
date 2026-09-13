# RAG 心理健康狀態分類

以 RAG（Retrieval-Augmented Generation）結合 LLM，對文字敘述進行心理健康狀態（status）分類，並比較「有／無 LLM 資料擴增」與「Prompt 最佳化前後」的效果。

## 檔案說明

| 檔案 | 用途 |
| --- | --- |
| `主程式.py` | 進入點，呼叫 `test.模型性能測試()` 執行評估 |
| `預處理.py` | 讀取原始 CSV、類別下採樣平衡、切分訓練／測試集 |
| `使用llm擴充資料.py` | 以 LLM 改寫（paraphrase）擴增訓練資料 |
| `建立RAG.py` | 以 SentenceTransformer 產生向量、建立 FAISS 索引 |
| `test.py` | 模型性能測試、混淆矩陣輸出 |
| `prompt最佳化.py` | Prompt 最佳化實驗 |

## 資料與索引（未納入版控）

`csv/` 與 `faiss/` 因體積過大（約 121MB）未上傳，需自行放置或重新產生：

```
csv/
  Combined Data.csv
  Combined_Data_Balanced.csv
  Combined_Data_Balanced_train_data.csv
  Combined_Data_Balanced_test_data.csv
faiss/
  llm資料擴增_faiss_index.bin
  llm資料擴增_rag_docs.json
  llm資料未擴增_faiss_index.bin
  llm資料未擴增_rag_docs.json
```

產生流程：`預處理.py` → `使用llm擴充資料.py` → `建立RAG.py` → `主程式.py`

## 環境需求

```bash
pip install -r requirements.txt
```

另需本機安裝並啟動 [Ollama](https://ollama.com/)。

## 注意

程式內目前使用絕對路徑（`/home/aiuser/wy/...`），在其他機器執行前需先調整。
