# Progress Update: Legal RAG Research Deep Dive

Completed web research deep dive on legal RAG provision-level retrieval/generation.

Files written:
- `research.md`

Key sources reviewed:
- Legal-DC / LegRAG: clause-level references, dual chunk+article retrieval, reranking, DSA, self-reflection.
- STARA: statutory tree parsing, lead-ins/chapeaus, definitions, cross-references, structured extraction; strong ablation evidence that statutory context beats prompt-only tweaks.
- ACORD: expert-graded clause retrieval, pointwise reranking, NDCG/precision@5, warning that overlap/weak labels are insufficient for legal reranking.

Main conclusion:
- The current project's bottleneck is now provision/card correctness, not source document retrieval and not generator ability in the abstract.
- Evidence cards are supported by statutory-research practice (STARA-like self-contained provisions), but they must be built from structured legal units and evaluated by slot metrics.

Recommended next technical direction:
1. Build structured legal-unit schema with parent/child relationships, clause lead, point text, article title.
2. Build card builder from structured fields, not raw text reparsing.
3. Add same-article point-deduction/suspension resolver.
4. Evaluate card-level fine/vehicle/citation/point-deduction accuracy before running generation.
5. Avoid prompt-only ablations and CE training from token-overlap labels.
