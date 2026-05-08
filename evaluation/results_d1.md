# RAG Retrieval Quality Evaluation: D1

## Setup
- **Corpus**: 7 internal reports (`data/reports/ko/`) → **34 chunks** (size=800 chars, overlap=100)
- **Embedding**: BGE-M3 (CPU, `normalize_embeddings=True`)
- **Vector store**: Chroma (`data/chroma_db_ko/`)
- **Retrieval**: hybrid (dense MMR `k=4 fetch_k=20` + BM25 `k=4`) fused via Reciprocal Rank Fusion (weights 0.5/0.5)

## Test queries (5 manual, simulating SHAP-driven driver lookups)
1. `12월 LA Port 컨테이너 피크시즌 영향`
2. `FBX 지수 상승이 컨테이너 운임에 주는 의미`
3. `PMI 50 돌파가 화물 수요에 미치는 영향`
4. `Trans-Pacific 해상 컨테이너 시장 점유율 경쟁사`
5. `GRI(General Rate Increase) 승인 절차`

## Results (precision@4, strict)

| Config | Q1 | Q2 | **Q3 (PMI)** | Q4 | Q5 | Total |
|---|---|---|---|---|---|---|
| Dense only (BGE-M3 + Chroma + MMR) | 3/4 | 2/4 | **0/4** ⚠️ | 2/4 | 2/4 | 9/20 (45%) |
| Hybrid (BM25 default whitespace split) | 4/4 | 2/4 | **0/4** ⚠️ | 3/4 | 2/4 | 11/20 (55%) |
| **Hybrid + Kiwi morpheme tokenizer** *(chosen)* | 3/4 | 1~2/4 | **1/4** ⭐ (PMI 챕터 등장) | 2/4 | 2/4 | 9~10/20 (45-50%) |

## Decision: Hybrid + Kiwi

### Rationale
1. **Q3 (PMI) chronic weakness partially resolved.** Kiwi splits Korean particles (`수요에` → `수요` + `에`), enabling BM25 to match the `### 3.2 China PMI` chapter in `external_drivers_handbook.md`. Dense-only and default-split BM25 both returned 0/4 on this query, meaning the LLM downstream would have had **zero domain context** for PMI and would likely hallucinate.
2. **Recall outranks precision in LLM downstream pipelines.** Solar-Pro can filter out distant chunks but cannot recover information that RAG never surfaces. Q1/Q4 saw minor precision regressions (~1 chunk each, often the same report's other chunk that the LLM would dedupe anyway), while Q3 saw a definitive recall gain.
3. **Web Search cannot fill this gap.** Internal-domain chapters (PMI as a leading indicator framing, GRI approval policy) don't exist on the public web. RAG must surface them; Tavily alone returns external indicator values without the firm's interpretation layer.

### Trade-offs accepted
- Kiwi splits Korean compound time terms (`12월` → `12` + `월`), introducing slight noise on month-specific queries. Monitor in D2-D3 evaluation; mitigation via `kiwi.add_user_word("12월")` if regression observed.
- Hybrid retrieval returns 5-7 candidates instead of strict 4 (EnsembleRetriever has no final-k cap). Downstream agent should slice top-k as needed.

### Backlog (deferred to D2-D3)
- **Query reformulation in agent's RAG node**: LLM rewrites SHAP driver name (`"pmi"`) into a fuller natural-language query (`"PMI 제조업 지수 선행지표 화물 수요"`). Expected to lift Q3 to 2-3/4.
- Frontmatter `route` normalization for metadata filtering (currently free text: `"global, applicable to all routes"` vs `"ICN-LAX"`). Postponed to D4-D5 report polish phase.
