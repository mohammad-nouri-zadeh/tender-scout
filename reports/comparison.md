# Comparison (same 1,000 test lots, March 2025)

| method                                                    |   macro_f1 |   accuracy |   usd_per_1000 | note                                          |
|:----------------------------------------------------------|-----------:|-----------:|---------------:|:----------------------------------------------|
| Majority class (always 33)                                |     0.0087 |     0.2421 |       0        | full test set                                 |
| TF-IDF + linear SVM (C=0.3)                               |     0.5097 |     0.745  |       0        | full test macro-F1 0.5735, unseen text 0.5651 |
| Embeddings amazon.titan-embed-text-v2:0 (512d) + LogReg   |     0.3385 |     0.611  |       0.012787 | trained on 20000 lots; 63934 tokens/1000      |
| TF-IDF + linear SVM, same training lots as the embeddings |     0.4173 |     0.693  |       0        | control: trained on the same 20000 lots       |
| Zero-shot LLM eu.amazon.nova-lite-v1:0                    |     0.2694 |     0.474  |       0.101596 | n=1000, 0 invalid, median 266.0 ms            |
