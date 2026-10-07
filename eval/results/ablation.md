# Ablation (7 October 2026, ollama:qwen3:8b, top 6 chunks)

Every configuration answered all 120 gold questions. These runs were measured before two later changes: the
16 large datasets were added to the index, and prose refusals started counting as "not found". Re-scoring the
saved answers under the new refusal rule turns 7 answers to unanswerable questions (summed over all ten runs) into
refusals and loses no correct answer;
the final measurement with both changes is in the README and in final_recommended.json / final_default.json.

**Held-out test split (80 questions)**

| Configuration | Evidence@1 | Evidence@6 | Answer row shown to the model | Answer accuracy (EN / AR) | Unanswerable refused | Wrong but shown | Withheld | Time per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 + rerank none | 57% | 86% | 86% | 78% (72% / 82%) | 90% | 6 | 6 | 3.7 s |
| bm25 + rerank rows | 67% | 87% | 87% | 75% (78% / 72%) | 90% | 9 | 8 | 3.9 s |
| bm25 + rerank llm | 73% | 87% | 87% | 75% (75% / 75%) | 90% | 8 | 8 | 6.8 s |
| bm25 + rerank none + 3 rows shown | 57% | 86% | 80% | 74% (75% / 72%) | 100% | 10 | 5 | 3.1 s |
| bm25 + rerank rows + 3 rows shown | 67% | 87% | 81% | 76% (80% / 72%) | 100% | 10 | 5 | 3.2 s |
| bm25 + rerank llm + 3 rows shown | 73% | 87% | 81% | 80% (82% / 78%) | 100% | 6 | 6 | 5.9 s |
| bm25 + query translation + rerank rows + 3 rows shown | 60% | 87% | 84% | 79% (75% / 82%) | 100% | 10 | 4 | 5.4 s |
| hybrid + rerank rows + 3 rows shown | 70% | 90% | 83% | 78% (85% / 70%) | 100% | 12 | 5 | 5.0 s |
| hybrid + query translation + rerank rows + 3 rows shown | 63% | 93% | 87% | 80% (82% / 78%) | 100% | 9 | 5 | 9.7 s |
| hybrid + query translation + rerank llm + 3 rows shown | 80% | 96% | 89% | 82% (80% / 85%) | 100% | 11 | 3 | 12.7 s |

**Dev split (40 questions, used for tuning)**

| Configuration | Evidence@1 | Evidence@6 | Answer row shown to the model | Answer accuracy (EN / AR) | Unanswerable refused | Wrong but shown | Withheld | Time per question |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25 + rerank none | 30% | 77% | 77% | 75% (70% / 80%) | 90% | 3 | 3 | 3.4 s |
| bm25 + rerank rows | 47% | 80% | 80% | 75% (70% / 80%) | 90% | 3 | 4 | 3.5 s |
| bm25 + rerank llm | 80% | 87% | 87% | 82% (80% / 85%) | 90% | 2 | 2 | 6.3 s |
| bm25 + rerank none + 3 rows shown | 30% | 77% | 73% | 80% (80% / 80%) | 100% | 4 | 1 | 3.1 s |
| bm25 + rerank rows + 3 rows shown | 47% | 80% | 77% | 78% (80% / 75%) | 100% | 5 | 2 | 3.1 s |
| bm25 + rerank llm + 3 rows shown | 80% | 87% | 80% | 82% (85% / 80%) | 100% | 3 | 2 | 5.8 s |
| bm25 + query translation + rerank rows + 3 rows shown | 53% | 83% | 80% | 80% (80% / 80%) | 100% | 5 | 1 | 5.5 s |
| hybrid + rerank rows + 3 rows shown | 60% | 90% | 83% | 78% (80% / 75%) | 100% | 7 | 1 | 5.0 s |
| hybrid + query translation + rerank rows + 3 rows shown | 63% | 97% | 90% | 85% (90% / 80%) | 90% | 3 | 2 | 9.7 s |
| hybrid + query translation + rerank llm + 3 rows shown | 77% | 93% | 90% | 88% (95% / 80%) | 90% | 2 | 2 | 12.5 s |
