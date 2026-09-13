import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import ollama
import json
# 需要安裝: pip install sentence-transformers faiss-cpu
from sentence_transformers import SentenceTransformer
import 十分之一_使用llm擴充資料

def 十分之一_建立RAG(path,model='all-MiniLM-L6-v2'):
    import faiss
    # 1) 載入 train_paraphrase_df (已含原文 + 改寫文)
    rag_df = pd.read_csv(path)
    # 2) 準備模型 & 建立向量
    embed_model = SentenceTransformer(f'{model}')

    # 先將每一筆 "statement"+" status" 合成一段文本 (或看你要不要只用 statement）
    # 這邊示範: "內容 label 是 狀態"
    rag_corpus = [
        f"{row['statement']} true_label is {row['status']}"
        for _, row in rag_df.iterrows()
    ]

    #print("\n開始建立向量資料庫 (Faiss index)...")
    corpus_embeddings = embed_model.encode(rag_corpus, show_progress_bar=True)
    corpus_embeddings = np.array(corpus_embeddings, dtype='float32')  # Faiss 需要 float32

    # 建立 Faiss index
    embedding_dim = corpus_embeddings.shape[1]  # e.g. 384 for 'all-MiniLM-L6-v2'
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(corpus_embeddings)

    # 可以建立一個對照表，存放實際的原文
    rag_docs = rag_corpus  # 與 embeddings 順序一致
    f_path = f"{path}"
    f_index_path = f"{f_path}_faiss_index.bin"
    f_docs_path = f"{f_path}_rag_docs.json"
    faiss.write_index(index, f"{f_index_path}")


    with open(f"{f_docs_path}", "w", encoding="utf-8") as f:
        json.dump(rag_docs, f, ensure_ascii=False, indent=4)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  
    import faiss

    # 載入已保存的 Faiss index
    index = faiss.read_index(f"{f_path}_faiss_index.bin")

    # 檢查資料庫中有多少筆資料
    #print(f"The Faiss index contains {index.ntotal} entries.")
    return f_index_path,f_docs_path