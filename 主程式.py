import test
index='/home/aiuser/wy/final/faiss/llm資料未擴增_faiss_index.bin'
docs='/home/aiuser/wy/final/faiss/llm資料未擴增_rag_docs.json'
test.模型性能測試(title="RAG(After Optimization)",f_index_path=index,f_docs_path=docs,frac=0.01,top_k=1)
# index='/final/faiss/llm資料擴增_faiss_index.bin'
# docs='/final/faiss/llm資料擴增_rag_docs.json'
# test.模型性能測試(title="RAG with Data Augmented(After Optimization)",f_index_path=index,f_docs_path=docs,frac=0.1,top_k=1)

