import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import csv
import faiss
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import ollama
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

def 模型性能測試(title, f_index_path, f_docs_path, frac=1, top_k=1):
    # === 1) 放大繪圖字體 ===
    matplotlib.rcParams.update({
        'font.size': 14,         # 整體字體大小
        'axes.titlesize': 18,    # 圖表標題字體大小
        'axes.labelsize': 16,    # x, y 軸標籤字體大小
        'xtick.labelsize': 14,   # x 軸刻度字體大小
        'ytick.labelsize': 14,   # y 軸刻度字體大小
        'legend.fontsize': 14,   # 圖例字體大小
        'figure.titlesize': 20   # figure 整體標題字體大小
    })

    # === (2) 讀取測試資料集 ===
    df = (
        pd.read_csv(f'final/csv/Combined_Data_Balanced_test_data.csv', usecols=['statement', 'status'])
        .sample(frac=frac, random_state=42)
    )
    test_df = df

    valid_labels = ['Normal', 'Depression', 'Anxiety', 'Bipolar']
    model_name = 'llama3.1'
    temperature = 0.1
    num_predict = 2000
    csv_file = "/final/csv/檢查用.csv"

    # 載入 SentenceTransformer
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    # embed_model = SentenceTransformer('all-MiniLM-L12-v2')

    def retrieve_reference_from_rag(query_text, index, rag_docs, top_k=3):
        """
        透過 Faiss 從 RAG 資料庫中檢索相似的參考資料
        """
        query_emb = embed_model.encode([query_text])[0].astype('float32')
        distances, indices = index.search(np.array([query_emb]), top_k)

        # 取出最相似的 top_k docs
        retrieved_texts = [rag_docs[idx] for idx in indices[0]]
        reference_context = "\n".join(retrieved_texts)
        return reference_context

    def classify_text_with_rag(text, index, rag_docs, model_name='llama3.1', valid_labels=None, top_k=1):
        """
        先到 RAG 資料庫中找相似的參考資料，再用 LLM 分類。
        """
        if valid_labels is None:
            valid_labels = []

        # step1: 先檢索
        reference_context = retrieve_reference_from_rag(text, index, rag_docs, top_k=top_k)

        # step2: LLM 分類
        prompt = f"""
Classify the text into one of {valid_labels}.
Below is some related reference content that might help you classify the new text:
{reference_context}

Now classify this text:
{text}

Please only output one of the following labels: {valid_labels}. Do not output anything else.
        """.strip()
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': ""}
        ]
        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={"temperature": temperature, "num_predict": num_predict}
            )
            result = response['message']['content'].strip()

            # 寫入 CSV (可選)
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([text, reference_context])

            # 檢查回應中是否包含任一有效標籤（不區分大小寫）
            for label in valid_labels:
                if label.lower() in result.lower():
                    return label
            # 若沒有找到有效標籤，回傳空字串，代表漏判
            return ""
        except Exception as e:
            # 若發生錯誤，回傳空字串
            return ""

    def classify_dataset_with_rag(dataset, index, rag_docs, model_name='llama3.1', valid_labels=None, top_k=3):
        if valid_labels is None:
            valid_labels = []

        y_true, y_pred, results = [], [], []

        for statement, status in tqdm(zip(dataset['statement'], dataset['status']), total=len(dataset)):
            predicted_label = classify_text_with_rag(
                statement,
                index,
                rag_docs,
                model_name=model_name,
                valid_labels=valid_labels,
                top_k=top_k
            )
            y_true.append(status)
            y_pred.append(predicted_label)
            results.append({
                "statement": statement,
                "true_label": status,
                "predicted_label": predicted_label
            })
        return y_true, y_pred, results

    def evaluate_and_plot_confusion_matrix(y_true, y_pred, valid_labels, top_k, title, save_path):
        """
        評估模型預測結果並繪製混淆矩陣。
        此處我們直接使用全部測試樣本 (包含空回應)，使 support 反映所有樣本，
        而空回應因不屬於 valid_labels 會導致該類的漏判。
        
        - y_true, y_pred: 分別是所有測試樣本的真實標籤與預測標籤
        - valid_labels: 有效標籤清單
        - top_k: top_k 參數值
        - title: 圖表標題
        - save_path: 混淆矩陣圖儲存路徑
        """
        # (A) 整體準確率 (空回應算作錯誤)
        test_accuracy = accuracy_score(y_true, y_pred)
        # (B) 統計空回應的數量
        empty_count = y_pred.count("")
        print(f"\n{title} - 總體準確率 (包含空回應): {test_accuracy:.4f}")
        print(f"{title} - 空回應的數量: {empty_count}")

        # (C) 利用全部測試資料計算混淆矩陣
        cm = confusion_matrix(y_true, y_pred, labels=valid_labels)

        # 各類別準確率 (對角線 / 該 row 總和)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        per_class_accuracy = np.nan_to_num(per_class_accuracy)
        print(f"\n{title} - 各類別的準確率：")
        for idx, acc in enumerate(per_class_accuracy):
            print(f"{valid_labels[idx]}: {acc:.2f}")

        print(f"{title} - top_k={top_k}")

        # (D) 繪製混淆矩陣
        plt.figure(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_labels)
        disp.plot(cmap="Blues", values_format='d', ax=plt.gca())
        ax = plt.gca()
        ax.set_xticklabels(valid_labels)
        ax.set_yticklabels(valid_labels)
        plt.title(title)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.xticks()
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        print(f"{title} - 混淆矩陣圖已保存為 '{save_path}'")

        # (E) 生成分類報告，利用全部測試資料（空回應作為漏判計入錯誤）
        print(f"\n===== Classification Report (全部樣本皆計入) - {title} =====")
        report = classification_report(y_true, y_pred, labels=valid_labels, zero_division=0, digits=4)
        print(report)

    # === (3) 進行測試集分類 ===
    # 載入 Faiss index
    index = faiss.read_index(f"{f_index_path}")
    # 載入對照表 (RAG文件)
    with open(f"{f_docs_path}", "r", encoding="utf-8") as f:
        rag_docs = json.load(f)

    # 執行分類
    y_true, y_pred, results = classify_dataset_with_rag(
        test_df,
        model_name=model_name,
        valid_labels=valid_labels,
        index=index,
        rag_docs=rag_docs,
        top_k=top_k
    )

    # 最後評估並繪製混淆矩陣、輸出指標
    evaluate_and_plot_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        valid_labels=valid_labels,
        top_k=top_k,
        title=title,
        save_path=f'{f_docs_path}_confusion_matrix.png'
    )
