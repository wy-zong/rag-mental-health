**Inference cost and runtime on the hardware reported in the implementation settings table**

| System                                  | Type                | Train cost               |   Median latency (ms/query) |   p95 latency (ms) |   Mean prompt tokens |   Total GPU sec |
|:----------------------------------------|:--------------------|:-------------------------|----------------------------:|-------------------:|---------------------:|----------------:|
| Llama 3.1 only                          | LLM (no training)   | none (no weight updates) |                      182.3  |              518.2 |                226.7 |           385.9 |
| + RAG                                   | LLM (no training)   | none (no weight updates) |                      312.2  |              891.6 |                427.6 |           654.9 |
| + RAG + augmentation                    | LLM (no training)   | none (no weight updates) |                      338.7  |              891.5 |                434.9 |           683.9 |
| + RAG + optimized prompt                | LLM (no training)   | none (no weight updates) |                      307    |              905.2 |                386.4 |           657.3 |
| + RAG + optimized prompt + augmentation | LLM (no training)   | none (no weight updates) |                      327.5  |              899   |                393.8 |           677.1 |
| tfidf_lr                                | supervised baseline | 6.7s                     |                        0.16 |              nan   |                nan   |             7   |
| minilm_lr                               | supervised baseline | 127.2s                   |                       13.66 |              nan   |                nan   |           150.4 |
| distilbert                              | supervised baseline | 10231.3s                 |                       83.94 |              nan   |                nan   |         10374   |