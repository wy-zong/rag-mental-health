import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import ollama
import json
from sentence_transformers import SentenceTransformer
import faiss
import csv
import matplotlib.pyplot as plt
import optuna
import prompt_100
import os
# ========== 讀取資料集 ==========

df = (
    pd.read_csv('/home/aiuser/wy/csv/Combined_Data_Balanced_test_data.csv', usecols=['statement', 'status'])
    .sample(frac=0.1, random_state=42)
)
test_df = df

valid_labels = ['Normal', 'Depression', 'Anxiety', 'Bipolar']
model_name = 'llama3.1'
temperature = 0.1
num_predict = 2000
csv_file = "/home/aiuser/wy/csv/檢查用.csv"

# Sentence-BERT 向量模型
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# embed_model = SentenceTransformer('all-MiniLM-L12-v2')

# ========== 讀取 FAISS Index 與 RAG Docs ==========

f_index_path = "/home/aiuser/wy/RAG/faiss/llm資料未擴增_faiss_index.bin"
f_docs_path = "/home/aiuser/wy/RAG/faiss/llm資料未擴增_rag_docs.json"

index = faiss.read_index(f_index_path)
with open(f_docs_path, "r", encoding="utf-8") as f:
    rag_docs = json.load(f)

# ========== 函式區 ==========

def retrieve_reference_from_rag(query_text, index, rag_docs, top_k=3):
    """
    透過 Faiss 從 RAG 資料庫中檢索相似的參考資料
    """
    query_emb = embed_model.encode([query_text])[0].astype('float32')
    distances, indices = index.search(np.array([query_emb]), top_k)
    
    retrieved_texts = []
    for idx in indices[0]:
        retrieved_texts.append(rag_docs[idx])
    reference_context = "\n".join(retrieved_texts)
    return reference_context


def classify_text_with_rag(
    text,
    index,
    rag_docs,
    model_name='llama3.1',
    valid_labels=None,
    top_k=1,
    prompt_template=None,
    temperature=0.1,
    num_predict=2000,
    csv_file="/home/aiuser/wy/csv/檢查用.csv",
):
    """
    使用 RAG + LLM 進行分類，可以帶入不同的 prompt_template。
    """
    if valid_labels is None:
        valid_labels = []
    
    # 1. RAG 檢索
    reference_context = retrieve_reference_from_rag(text, index, rag_docs, top_k=top_k)
    
    # 2. 組合 Prompt（這裡可自由改寫 prompt_template）
    #    注意：prompt_template 中預留 {valid_labels}、{reference_context}、{input_text} 供 format 替換
    prompt = prompt_template.format(
        valid_labels=valid_labels,
        reference_context=reference_context,
        text=text
    )
    
    messages = [
        {'role': 'system', 'content': prompt},
        {'role': 'user', 'content': ""}
    ]

    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"temperature": temperature, "num_predict": num_predict},
        )
        result = response['message']['content'].strip()
        
        # 將檢索到的內容寫入 CSV（依需求可自行調整要寫入什麼）
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([text, reference_context])
        
        # 檢查是否有在 valid_labels 內
        for label in valid_labels:
            if label.lower() in result.lower():
                return label
        return None
    except Exception as e:
        return None


def classify_dataset_with_rag(dataset, index, rag_docs, model_name, valid_labels, top_k, prompt_template):
    """
    針對整個 dataset 做分類。
    """
    y_true, y_pred, results = [], [], []
    
    for statement, status in tqdm(zip(dataset['statement'], dataset['status']), total=len(dataset)):
        predicted_label = classify_text_with_rag(
            text=statement,
            index=index,
            rag_docs=rag_docs,
            model_name=model_name,
            valid_labels=valid_labels,
            top_k=top_k,
            prompt_template=prompt_template
        )
        
        y_true.append(status)
        y_pred.append(predicted_label)
        results.append({
            "statement": statement,
            "true_label": status,
            "predicted_label": predicted_label
        })
    
    return y_true, y_pred, results


def evaluate_confusion_matrix(y_true, y_pred, valid_labels, top_k, title, save_path):
    """
    評估預測結果並繪製混淆矩陣。
    """
    # 把 None 視為 INVALID
    pairs = [(t, p if p is not None else "INVALID") for t, p in zip(y_true, y_pred)]
    if len(pairs) == 0:
        print(f"\n{title} - 無法計算，因為沒有任何預測結果。")
        return
    
    y_true_filtered, y_pred_filtered = zip(*pairs)
    test_accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    invalid_count = y_pred_filtered.count("INVALID")
    
    print(f"\n{title} - 總體準確率: {test_accuracy:.2f}")
    print(f"{title} - INVALID 的數量: {invalid_count}")
    
    # 計算混淆矩陣（包含 INVALID）
    all_labels = valid_labels + ["INVALID"]
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=all_labels)
    
    # 計算每個類別的準確率（若該類別沒有真實樣本，則跳過 NaN）
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    per_class_accuracy = np.nan_to_num(per_class_accuracy)
    
    print(f"\n{title} - 各類別的準確率：")
    for idx, label in enumerate(all_labels):
        print(f"{label}: {per_class_accuracy[idx]:.2f}")
    
    print(f"{title} - top_k={top_k}")
    plt.rcParams.update({'font.size': 15})
    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(cmap="Blues", values_format='d', ax=plt.gca())
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"{title} - 混淆矩陣圖已保存為 '{save_path}'")


# ========== Optuna 相關 ==========

# 用來收集每次 trial 中，準確率低於某一閾值 (0.75) 的 prompt
low_accuracy_prompts = []

def objective(trial):
    """
    Optuna 的目標函式。會在這裡嘗試不同的 Prompt (或其他超參數)。
    然後跑完整個推論流程，回傳 Accuracy 給 Optuna 進行最佳化。
    """
    # (1) 定義你想要測試的 prompt
    prompt_choice = trial.suggest_categorical(
        "prompt_template",
        [

    """
    Classify the text into one of {valid_labels}.
    Below is some related reference content that might help you classify the new text:
    {reference_context}

    Now classify this text:
    {text}

    Please only output one of the following labels: {valid_labels}. Do not output anything else.
    """,
        """Classify the text into one of {valid_labels}.
    Below is some related reference material that may help:
    {reference_context}

    Now, please look at the text:
    {text}

    Please only output one of the following labels: {valid_labels}. Do not output anything else.""",
        """You have the following reference to help you classify the text:
    {reference_context}

    The text to classify is:
    {text}

    Select exactly one label from {valid_labels} and output it. Do not output anything else.""",
        """Based on the reference below:
    {reference_context}

    Categorize this text:
    {text}

    Only choose one label from {valid_labels}. Output just that label.""",
        """Read the reference context here:
    {reference_context}

    Then read the text:
    {text}

    Determine the single label from {valid_labels} that best fits. Do not include anything besides that label.""",
        """Refer to this context:
    {reference_context}

    Classify the following text:
    {text}

    Provide exactly one label from {valid_labels}, and do not add any explanation.""",
        """Check the reference material here:
    {reference_context}

    Now determine the label for the text:
    {text}

    Your answer must be a single label from {valid_labels}. Output nothing else.""",
        """Use the reference content:
    {reference_context}

    Decide which label from {valid_labels} matches this text:
    {text}

    Output exactly one label from {valid_labels}, without further explanation.""",
        """Please classify this text into the correct label among {valid_labels}.
    Supporting information:
    {reference_context}

    Text:
    {text}

    Output only one label from {valid_labels} and nothing else.""",
        """Here is your reference data:
    {reference_context}

    Now, classify the text:
    {text}

    Answer with a single label from {valid_labels}, and do not elaborate.""",
        """The reference below should help:
    {reference_context}

    Identify the correct label for the text:
    {text}

    Output only one from {valid_labels}.""",
        """Analyze the reference context:
    {reference_context}

    Then read the text:
    {text}

    Choose the most fitting label from {valid_labels} and output it only.""",
        """Here's some reference:
    {reference_context}

    Classify this text:
    {text}

    Please return only one label from {valid_labels}. No additional output.""",
        """Below is relevant reference information:
    {reference_context}

    Given the text:
    {text}

    Select the single best match from {valid_labels}, and output only that label.""",
        """We have reference information here:
    {reference_context}

    Consider the text carefully:
    {text}

    Please choose the correct category from {valid_labels}, and provide only that label.""",
        """Below you'll find reference information:
    {reference_context}

    Based on it, categorize this text:
    {text}

    Only respond with the label from {valid_labels} that best applies.""",
        """Use the reference content:
    {reference_context}

    Then determine which of {valid_labels} applies to this text:
    {text}

    Output just the appropriate label.""",
        """We have a set of valid labels: {valid_labels}.
    Reference data:
    {reference_context}

    Now read the text:
    {text}

    Output the single correct label. Nothing else.""",
        """Examine the context:
    {reference_context}

    Decide which one of {valid_labels} applies to this text:
    {text}

    Provide only the most suitable label.""",
        """This is the reference context:
    {reference_context}

    Here is the text to classify:
    {text}

    Select the best label from {valid_labels}, with no extra text.""",
        """You have the following valid labels: {valid_labels}.
    Consider this reference:
    {reference_context}

    Now, classify the text:
    {text}

    Output a single label from the list and nothing else.""",
        """Refer to the provided context:
    {reference_context}

    Identify the single most fitting label from {valid_labels} for this text:
    {text}

    Output only that label.""",
        """Task: pick one label out of {valid_labels}.
    Reference info:
    {reference_context}

    Here is the text:
    {text}

    Please respond with only the most appropriate label.""",
        """This reference may guide you:
    {reference_context}

    Given this text:
    {text}

    Please choose one label from {valid_labels}. Nothing else.""",
        """Consult the reference below:
    {reference_context}

    Based on it, classify the text:
    {text}

    Respond with one label from {valid_labels}.""",
        """A reference is provided:
    {reference_context}

    Determine which label in {valid_labels} fits the text:
    {text}

    Output only that single label.""",
        """The following context is provided:
    {reference_context}

    Please classify the text:
    {text}

    Pick the best fitting label from {valid_labels} and output only that label.""",
        """Here is relevant reference information:
    {reference_context}

    Evaluate the text:
    {text}

    Select exactly one label from {valid_labels} to describe it. Provide no further commentary.""",
        """Look at the reference context:
    {reference_context}

    The text to categorize:
    {text}

    You must choose one label from {valid_labels}. Nothing else should be output.""",
        """Below is some reference content:
    {reference_context}

    The given text is:
    {text}

    Choose one label out of {valid_labels} and output only that.""",
        """Please choose the correct label among {valid_labels}.
    Reference:
    {reference_context}

    Text:
    {text}

    Respond with the single label that best applies. Nothing else.""",
        """Look at this reference context:
    {reference_context}

    Then classify the text as one of {valid_labels}:
    {text}

    Provide only that label as your output.""",
        """Here is your reference:
    {reference_context}

    Determine which single label from {valid_labels} matches the text:
    {text}

    Only provide that label, with no additional words.""",
        """Check the provided reference:
    {reference_context}

    Then choose the correct label from {valid_labels} for the text:
    {text}

    Output only the chosen label.""",
        """Refer to:
    {reference_context}

    Classify the text into one label from {valid_labels}:
    {text}

    Output only that one label.""",
        """Using the reference data below:
    {reference_context}

    Identify the correct label among {valid_labels} for this text:
    {text}

    Provide only the single label.""",
        """Here is the reference for you to consider:
    {reference_context}

    Now label this text:
    {text}

    You must select one among {valid_labels}. Output only that label.""",
        """Below reference might help:
    {reference_context}

    The text to be labeled is:
    {text}

    Choose exactly one label from {valid_labels}. No other text.""",
        """This is the reference data:
    {reference_context}

    Analyze and label this text:
    {text}

    Provide just the single label from {valid_labels}.""",
        """Review the following reference:
    {reference_context}

    Then choose which one of {valid_labels} fits the text:
    {text}

    Only return that label as your answer.""",
        """Below is some reference information for classification:
    {reference_context}

    The text is:
    {text}

    Pick the single correct label from {valid_labels}. Respond with that label only.""",
        """Here is the reference:
    {reference_context}

    Please identify the suitable label among {valid_labels} for the text:
    {text}

    Only state the chosen label, nothing else.""",
        """Use the reference context:
    {reference_context}

    Select the appropriate label from {valid_labels} for the text:
    {text}

    Output the single label only.""",
        """These are the valid labels: {valid_labels}.
    Reference is:
    {reference_context}

    Given the text:
    {text}

    Provide just one label from {valid_labels}.""",
        """We have reference context here:
    {reference_context}

    The text to label is:
    {text}

    Select one from {valid_labels}, and provide only that label.""",
        """Take the reference below into account:
    {reference_context}

    Then pick the best fitting label from {valid_labels} for:
    {text}

    Output only that one label.""",
        """There's reference data here:
    {reference_context}

    Classify this text:
    {text}

    Use exactly one label from {valid_labels} as your answer. Nothing else.""",
        """Refer to this data for guidance:
    {reference_context}

    Then assign one of the labels in {valid_labels} to this text:
    {text}

    Only return the single chosen label.""",
        """We have this reference for you:
    {reference_context}

    Look at the text:
    {text}

    Decide on a single label from {valid_labels}. Output nothing but that label.""",
        """Classify the following text using the reference:
    {reference_context}

    Text:
    {text}

    Choose only one label from {valid_labels}. Provide no other information.""",
        """Reference context:
    {reference_context}

    The text is:
    {text}

    From the set {valid_labels}, pick the single best label. Output it alone.""",
        """Use this reference:
    {reference_context}

    Decide the correct label from {valid_labels} for:
    {text}

    Provide only that label in your response.""",
        """The reference context is here:
    {reference_context}

    Determine which label from {valid_labels} applies to:
    {text}

    Provide only the single label. No extra text.""",
        """There is reference context to help:
    {reference_context}

    Categorize the text:
    {text}

    The valid labels are {valid_labels}. Output a single label and nothing more.""",
        """Below is the reference context:
    {reference_context}

    Indicate which one of {valid_labels} fits this text best:
    {text}

    Only provide that label and nothing else.""",
        """Given the reference context:
    {reference_context}

    Classify the text:
    {text}

    Please only output one of the following labels: {valid_labels}. Do not output anything else.""",
        """Review the information in {reference_context}.
Then read this text: {text}.
Choose the single most fitting label from {valid_labels} and output only that label.""",
        """Analyze the reference context here: {reference_context}.
Which label from {valid_labels} best describes the text below?
{text}
Please provide only the chosen label.""",
        """Refer to {reference_context} as your guide.
The text is: {text}.
Identify one suitable label from {valid_labels}, and give that label only.""",
        """We have some reference material: {reference_context}.
Based on it, categorize: {text}.
From {valid_labels}, select the one correct label. Output just that.""",
        """Consult {reference_context} for guidance.
Determine the best matching label in {valid_labels} for:
{text}
Your output should be that label alone.""",
        """Here is reference data to help classify: {reference_context}.
The text reads: {text}.
Pick only one label from {valid_labels}, and output it plainly.""",
        """Reference: {reference_context}.
Which one of {valid_labels} fits the text below?
{text}
Supply only the chosen label.""",
        """With the reference content {reference_context},
evaluate the text: {text}.
Output the single correct label from {valid_labels}, with no extra text.""",
        """Look at the provided reference: {reference_context}.
Given this text: {text},
pick the most appropriate label from {valid_labels}, returning only that.""",
        """Given {reference_context},
decide which label in {valid_labels} applies to the text:
{text}
Respond with just that label.""",
        """Use the following reference: {reference_context}.
Determine the correct single label from {valid_labels} for this text: {text}.
No other output, please.""",
        """Here's some reference data: {reference_context}.
We need to classify this text: {text}.
Select the single applicable label from {valid_labels} and output only that.""",
        """Consult {reference_context} for classification guidance.
Identify which label in {valid_labels} fits:
{text}
Please respond with only that label.""",
        """Reference info: {reference_context}.
Look at the text: {text}.
Which of {valid_labels} best describes it? Provide only that label.""",
        """Here's the reference: {reference_context}.
Given the text: {text},
which single label from {valid_labels} fits? Output it alone.""",
        """We have the context below: {reference_context}.
Examine the text: {text}.
Choose the one most appropriate label from {valid_labels}. Provide only that label.""",
        """We have some details in {reference_context}.
Given the text: {text},
select only one label from {valid_labels}. Output nothing else.""",
        """Reference context: {reference_context}.
Text in question: {text}.
Which label from {valid_labels} matches? Output that label only.""",
        """Here's the context: {reference_context}.
Please identify which one of {valid_labels} applies to:
{text}
Only output the chosen label.""",
        """Context: {reference_context}
Text: {text}
From the set {valid_labels}, choose only one appropriate label. Nothing else.""",
        """Take note of {reference_context}.
Given the text: {text},
which label in {valid_labels} fits best? Provide only that label.""",
        """Using the context: {reference_context},
classify the text: {text} under one of these: {valid_labels}.
Your response should be only that one label.""",
        """Check the classification details in {reference_context}.
For {text},
pick the single label from {valid_labels} that best fits. Output it alone.""",
        """Reference context: {reference_context}.
Text: {text}.
Please identify the correct label from {valid_labels}. Only the label, nothing else.""",
        """Look at the provided details in {reference_context}.
Decide which of {valid_labels} best categorizes this text: {text}.
Give only that category.""",
        """According to {reference_context},
choose the right label from {valid_labels} for the text: {text}.
Respond with only the chosen label.""",
        """Read the context: {reference_context}.
Text to classify: {text}.
Pick the single correct option in {valid_labels}. Provide no other text.""",
        """Context available: {reference_context}.
Now identify the label from {valid_labels} for: {text}.
Return just that label.""",
        """Look over {reference_context}.
Apply it to this text: {text}.
Which one of {valid_labels} fits? Provide only that label.""",
        """Reference content: {reference_context}.
Text in question: {text}.
Select only one label from {valid_labels} and provide it with no additional text.""",
        """Evaluate the text: {text} using the reference: {reference_context}.
Which single label from {valid_labels} is correct?
Please provide only that label.""",
        """We have reference data: {reference_context}.
Based on it, classify the text: {text}.
Only one label from {valid_labels}. No extra words.""",
        """Classification reference: {reference_context}.
The text is {text}.
Select the correct label from {valid_labels} and provide only that.""",
        """Here's your reference: {reference_context}.
Use it to label: {text}.
Only a single label from {valid_labels} is allowed. Output nothing else.""",
        """Use the guidelines in {reference_context}.
Review the text: {text}.
Which of {valid_labels} matches? Respond with that single label only.""",
        """We have helpful reference context: {reference_context}.
Read the text: {text}.
Determine the single suitable label from {valid_labels}. Output that alone.""",
        """Rely on the data in {reference_context}.
Here's the text: {text}.
Which label from {valid_labels} is correct? Output only that.""",
        """Use {reference_context}.
For the text: {text},
choose one among {valid_labels}. Please respond with only that label.""",
        """We have reference info in {reference_context}.
Evaluate the text: {text}.
Determine which label of {valid_labels} applies. Output only the label.""",
        """Utilize the reference content: {reference_context}.
Determine the appropriate label from {valid_labels} for {text}.
Only that label should be in your reply.""",
        """Refer to {reference_context} for classification help.
Given {text},
pick the correct label from {valid_labels}. Output it without explanation.""",
        """Classification reference: {reference_context}.
We have the text: {text}.
Select only one label from {valid_labels}, return it, no extra words.""",
        """We've got the reference: {reference_context}.
Identify the correct label from {valid_labels} for {text}.
Reply solely with that label.""",
        """Use details in {reference_context} to determine the label.
Text: {text}
Pick one label from {valid_labels}. Output that alone.""",
        """We have the following text: {text}.
Given {reference_context},
select one label from {valid_labels} that best fits. Output only that label.""",
        """Classification guide: {reference_context}.
Please apply it to the text: {text}.
Which of {valid_labels} is correct? Provide only that.""",
        """Assess the text: {text} with help from {reference_context}.
Pick precisely one label from {valid_labels} as your answer. Nothing else.""",
        """We have the text: {text}.
Reference: {reference_context}.
Which label in {valid_labels} fits best? Return that label alone.""",
        """Look at {reference_context}.
Now read this text: {text}.
Select the one label from {valid_labels} that applies. Respond only with that label.""",
        """Consult the reference in {reference_context}.
Text: {text}.
Choose exactly one label from {valid_labels} with no extra commentary.""",
        """Given classification guide: {reference_context}.
For the text: {text},
pick the label from {valid_labels} that best fits. Provide it by itself.""",
        """We have the following reference: {reference_context}.
Classify the text: {text}.
Use a single label from {valid_labels}. Nothing else in your reply.""",
        """According to {reference_context},
assign one label from {valid_labels} to {text}.
Output the label alone, with no further text.""",
        """Consult the provided reference: {reference_context}.
Look at {text}.
Which single label from {valid_labels} is correct? Respond with only that.""",
        """Classification references: {reference_context}.
Text: {text}.
Please respond with one suitable label from {valid_labels}, nothing else.""",
        """We have the text: {text}.
Reference data: {reference_context}.
Which label in {valid_labels} fits? Provide only that label.""",
        """Based on reference: {reference_context},
select the suitable label from {valid_labels} for:
{text}
Return just that single label.""",
        """Check out the classification reference: {reference_context}.
Considering the text: {text},
which one from {valid_labels} is correct? Respond solely with that label.""",
        """Classification context: {reference_context}.
Which single label from {valid_labels} matches this text?
{text}
Respond with only that label.""",
        """Reference data: {reference_context}.
Text to classify: {text}.
Return only the matching label from {valid_labels}, with no extra text.""",
        """Check the context: {reference_context}.
Look at the text: {text}.
Which label among {valid_labels} applies? Just give that label.""",
        """We have the reference context {reference_context}.
For the text: {text},
select the single valid label from {valid_labels}. Output that alone.""",
        """Context for classification is {reference_context}.
The text to evaluate is {text}.
From {valid_labels}, return the one correct label. Nothing else.""",
        """We have reference content: {reference_context}.
For the text below: {text},
pick only one label from {valid_labels}. Nothing else is needed.""",
        """Classification list: {valid_labels}.
Reference provided: {reference_context}.
Given {text}, select a single label from the list. Output only that label.""",
        """Look at the reference context: {reference_context}.
Then decide on a label from {valid_labels} for {text}.
Provide only that one label.""",
    ]

    )
    
    # (2) 也可以同時測試 top_k (這裡固定為 1)
    top_k = 1
    
    # (3) 執行推論並計算 Accuracy
    y_true, y_pred, _ = classify_dataset_with_rag(
        dataset=test_df,
        index=index,
        rag_docs=rag_docs,
        model_name=model_name,
        valid_labels=valid_labels,
        top_k=top_k,
        prompt_template=prompt_choice
    )
    
    # (4) 計算本次參數組合的 accuracy
    pairs = [(t, p if p is not None else "INVALID") for t, p in zip(y_true, y_pred)]
    y_true_filtered, y_pred_filtered = zip(*pairs)
    test_accuracy = accuracy_score(y_true_filtered, y_pred_filtered)

    # 若準確率低於 0.75，則將該 prompt_template、trial 編號、accuracy 記錄下來
    if test_accuracy < 0.75:
        low_accuracy_prompts.append({
            "trial": trial.number,
            "prompt_template": prompt_choice,
            "accuracy": test_accuracy
        })
    
    # (5) 回傳給 Optuna 進行最佳化
    return test_accuracy


if __name__ == "__main__":
    # 建立 Optuna Study
    study = optuna.create_study(
        direction="maximize",  # 我們想要最大化 accuracy
        sampler=optuna.samplers.TPESampler(seed=42)  # 設定隨機種子以利重現
    )
    
    # 進行超參數搜尋，根據需求調整 n_trials
    study.optimize(objective, n_trials=250)
    
    # 顯示最佳結果
    best_trial = study.best_trial
    print("\n========== Optuna 結果 ==========")
    print("Best trial ID:", best_trial.number)
    print("Best trial value (Accuracy):", best_trial.value)
    print("Best trial params:", best_trial.params)
    
    # ========== 用最佳 Prompt 重新做一次完整預測 + 混淆矩陣可視化 ==========

    # ========== 用最佳 Prompt 重新做一次完整預測 + 混淆矩陣可視化 ==========
    best_prompt = best_trial.params["prompt_template"]
    best_top_k = 1  # 若有動態測試 top_k，這裡則換成 best_trial.params["top_k"]

    y_true, y_pred, _ = classify_dataset_with_rag(
        dataset=test_df,
        index=index,
        rag_docs=rag_docs,
        model_name=model_name,
        valid_labels=valid_labels,
        top_k=best_top_k,
        prompt_template=best_prompt
    )

    # 這裡先算一次混淆矩陣需要的東西
    pairs = [(t, p if p is not None else "INVALID") for t, p in zip(y_true, y_pred)]
    y_true_filtered, y_pred_filtered = zip(*pairs)
    all_labels = valid_labels + ["INVALID"]

    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=all_labels)
    test_accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    per_class_accuracy = np.nan_to_num(per_class_accuracy)

    title = "Best Prompt with Optuna"
    print(f"\n{title} - 總體準確率: {test_accuracy:.2f}")
    print(f"{title} - INVALID 的數量: {y_pred_filtered.count('INVALID')}")

    print(f"\n{title} - 各類別的準確率：")
    for idx, label in enumerate(all_labels):
        print(f"{label}: {per_class_accuracy[idx]:.2f}")

    print(f"{title} - top_k={best_top_k}")

    # 開始畫混淆矩陣
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
    disp.plot(cmap="Blues", values_format='d', ax=plt.gca())
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.ylabel("True Label", fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # 存檔路徑
    save_folder = "/home/aiuser/wy/picture"
    cm_img_path = os.path.join(save_folder, "best_confusion_matrix.png")
    plt.savefig(cm_img_path, bbox_inches='tight')
    plt.show()
    print(f"{title} - 混淆矩陣圖已保存為 '{cm_img_path}'")

    # ========== Optuna Learning Curve 圖 ==========
    trials = study.get_trials()
    scores = [t.value for t in trials]  # 每個 trial 的 Accuracy

    # 計算「best so far」數據
    best_scores = []
    best_so_far = float('-inf')
    for score in scores:
        best_so_far = max(best_so_far, score)
        best_scores.append(best_so_far)

    # 繪製並儲存
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(12, 7.5))
    plt.plot(range(1, len(best_scores) + 1), best_scores, marker='o', label="Best so far")
    plt.title("Learning Curve - Best So Far")
    plt.xlabel("Trial")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    curve_img_path = os.path.join(save_folder, "optuna_learning_curve_best.png")
    plt.savefig(curve_img_path)
    plt.close()
    print(f"Learning Curve 圖已保存為 '{curve_img_path}'")

    # 再畫一次 optuna 的預設可視化
    optuna.visualization.matplotlib.plot_optimization_history(study)
    optuna_img_path = os.path.join(save_folder, "optuna_optimization_history.png")
    plt.savefig(optuna_img_path)
    plt.close()
    print(f"Optuna Optimization History 圖已保存為 '{optuna_img_path}'")

    # ========== 將低準確率 (<0.75) 的 prompt_templates 寫入 CSV (原程式碼) ==========
    if low_accuracy_prompts:
        csv_output_path = "/home/aiuser/wy/csv/low_accuracy_prompts.csv"
        with open(csv_output_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["trial", "prompt_template", "accuracy"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in low_accuracy_prompts:
                writer.writerow(row)
        print(f"\n==== 共有 {len(low_accuracy_prompts)} 筆低於 0.75 的紀錄，已寫入 '{csv_output_path}' ====\n")
    else:
        print("\n==== 沒有任何低於 0.75 的 Prompt 記錄 ====\n")

    # ========== [新加入] 把「繪圖需要用到」的資料都存起來 ==========
    import numpy as np

    output_folder = "/home/aiuser/wy/output_data"
    os.makedirs(output_folder, exist_ok=True)

    # 1. 混淆矩陣資料存檔
    #    包含 cm、all_labels、best_top_k、test_accuracy、per_class_accuracy等等
    np.savez(
        os.path.join(output_folder, "confusion_matrix_data.npz"),
        cm=cm,
        all_labels=np.array(all_labels, dtype=object),
        test_accuracy=test_accuracy,
        per_class_accuracy=per_class_accuracy,
        best_top_k=best_top_k
    )

    # 2. Learning Curve 資料存檔
    #    包含 scores (每次 trial 的 accuracy) 和 best_scores (best so far)
    np.savez(
        os.path.join(output_folder, "optuna_learning_curve_data.npz"),
        scores=scores,
        best_scores=best_scores
    )

    print("\n>>> 已將混淆矩陣 + Learning Curve 資料以 npz 格式存到 output_data 資料夾！")