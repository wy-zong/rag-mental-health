import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ollama
import faiss

def 使用llm擴充資料(df_csv_path,frac=1,num_of_paraphrases=2,temperature=np.arange(1)):
    # ========== (3) 定義「單次產生 paraphrase」的函式 ==========

    def generate_paraphrase_once(text, emotion, model_name='llama3.1',temperature=0.7):
        """
        使用 LLaMA 生成單一改寫版本。
        不論模型回傳內容如何（可能多行），
        這裡都當作一段文本來使用。
        """
        prompt_for_paraphrase = f"""
    Please rewrite the following text with {emotion} emotion into 1 different version.
    Keep the original meaning and emotion, and do not add extra explanation.
    Text: {text}
    """
        messages = [
            {'role': 'system', 'content': prompt_for_paraphrase},
            {'role': 'user', 'content': ""}
        ]
        try:
            response = ollama.chat(
                model=model_name,
                messages=messages,
                options={"temperature": 0.7}
            )
            raw_text = response['message']['content'].strip()
            
            # 如擔心多行，可只取第一行。
            # paraphrased_version = raw_text.split('\n', 1)[0].strip()
            # 這裡先示範直接整段使用
            paraphrased_version = raw_text
            
            return paraphrased_version
        except Exception as e:
            #print(f"Paraphrasing Error: {e}")
            return ""


    # ========== (4) 產生「訓練集」參考資料 (含原文本 + 改寫文本) ==========
    # num_of_paraphrases = 2  # 想要改寫的次數，預設2
    # temperature_list = (1)
    # frac = 1
    df = (
        pd.read_csv(df_csv_path, usecols=['statement', 'status'])
        .sample(frac=frac, random_state=42)
    )
    train_df = df
    
    #print("\n開始為訓練集生成改寫文本...")

    train_paraphrase_records = []

    #for statement, status in tqdm(zip(train_df['statement'], train_df['status']), total=len(train_df)):
    for statement, status in zip(train_df['statement'], train_df['status']):   
        # 先將原始文本加入
        train_paraphrase_records.append({"statement": statement, "status": status})
        # 根據需求呼叫多次
        for _ in range(num_of_paraphrases):
            paraphrase_text = generate_paraphrase_once(statement, status, model_name='llama3.1', temperature=temperature)
            
            if paraphrase_text:  # 成功產生才加入
                train_paraphrase_records.append({"statement": paraphrase_text, "status": status})
                

    # 建立新的 DataFrame
    train_paraphrase_df = pd.DataFrame(train_paraphrase_records)
    #print(f"\n成功產生 {len(train_paraphrase_df)} 筆 (文本, 情緒) 訓練資料（包含原文和改寫）。")

    # 將所有參考資料寫入 CSV 檔
    train_paraphrase_csv_path = f'{df_csv_path}_f={frac}.n={num_of_paraphrases}.t={temperature}_train_paraphrases.csv'
    train_paraphrase_df.to_csv(train_paraphrase_csv_path, index=False)
    #print(f"本次使用的temperature: {temperature}")
    #print(f"產生的訓練參考資料已儲存至: {train_paraphrase_csv_path}")
    return train_paraphrase_csv_path,frac,num_of_paraphrases,temperature

