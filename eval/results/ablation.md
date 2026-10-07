# Reranker ablation (2026-10-07, ollama:qwen3:8b, bm25 retrieval, top 6)

**Held-out test split (80 questions)**

| Reranker | Evidence@1 | Evidence@6 | Dataset hit@1 | Answer accuracy (EN / AR) | Unanswerable refused | Wrong but shown | Withheld |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 57% | 86% | 70% | 78% (72% / 82%) | 90% | 6 | 6 |
| rows | 67% | 87% | 76% | 75% (78% / 72%) | 90% | 9 | 8 |
| llm | 73% | 87% | 80% | 75% (75% / 75%) | 90% | 8 | 8 |

**Dev split (40 questions, used for tuning)**

| Reranker | Evidence@1 | Evidence@6 | Dataset hit@1 | Answer accuracy (EN / AR) | Unanswerable refused | Wrong but shown | Withheld |
| --- | --- | --- | --- | --- | --- | --- | --- |
| none | 30% | 77% | 63% | 75% (70% / 80%) | 90% | 3 | 3 |
| rows | 47% | 80% | 67% | 75% (70% / 80%) | 90% | 3 | 4 |
| llm | 80% | 87% | 87% | 82% (80% / 85%) | 90% | 2 | 2 |
