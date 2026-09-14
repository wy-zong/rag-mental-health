**Consolidated results across the five conditions (accuracy counts invalid responses as incorrect)**

| Condition                               |   Accuracy |   Macro-F1 |   Weighted-F1 |   Invalid (n) |   Invalid (%) |    N |   Δ Accuracy vs prev (pp) | 95% CI of Δ (pp)   |   Holm p |
|:----------------------------------------|-----------:|-----------:|--------------:|--------------:|--------------:|-----:|--------------------------:|:-------------------|---------:|
| Llama 3.1 only                          |     0.7382 |     0.7401 |        0.7401 |            14 |          0.82 | 1700 |                   nan     |                    | nan      |
| + RAG                                   |     0.8182 |     0.8193 |        0.8193 |             3 |          0.18 | 1700 |                     8     | [+6.00, +10.00]    |   0      |
| + RAG + augmentation                    |     0.8165 |     0.8182 |        0.8182 |             6 |          0.35 | 1700 |                    -0.176 | [-1.06, +0.71]     |   1      |
| + RAG + optimized prompt                |     0.7924 |     0.7951 |        0.7951 |             7 |          0.41 | 1700 |                    -2.588 | [-3.88, -1.29]     |   0.0003 |
| + RAG + optimized prompt + augmentation |     0.7929 |     0.7958 |        0.7958 |             8 |          0.47 | 1700 |                     5.471 | [+3.24, +7.71]     |   0      |