# CRAG for Health Claim Verification

A re-implementation of **Corrective Retrieval-Augmented Generation (CRAG)** ([Yan et al., 2024](https://arxiv.org/abs/2401.15884))
for biomedical fact-checking. The system labels a health claim as **true** or **false**. We evaluate it on
PubHealth and compare it against an LLM with no retrieval and against standard RAG.

> 42173 Advanced Natural Language Processing, UTS, Spring 2026

Standard RAG trusts whatever the retriever returns. CRAG adds a lightweight **retrieval evaluator** that scores each
retrieved passage. Based on those scores it takes one of three actions:

| Action | Trigger | Knowledge given to the generator |
|---|---|---|
| **Correct** | at least one passage scores ≥ `UPPER` | retrieved passages, refined into their most relevant strips |
| **Incorrect** | every passage scores < `LOWER` | retrieved passages discarded; web search results instead |
| **Ambiguous** | otherwise | refined strips + web search results |

```
claim ──► Contriever retriever ──► T5 retrieval evaluator ──► action trigger
          (health Wikipedia)       (score per passage)           │
                                         ┌───────────────────────┼────────────────────────┐
                                     Correct                 Ambiguous                Incorrect
                                  refine strips        refine + rewrite + search    rewrite + web search
                                         └───────────────────────┼────────────────────────┘
                                                                 ▼
                                                   Qwen2.5-Instruct generator ──► true / false
```

---

## Implementation

The pipeline lives in [src/crag/](src/crag/). Each file is one CRAG stage, and
[pipeline.py](src/crag/pipeline.py) wires them together.

### 1. Data — [data.py](src/crag/data.py)

- **PubMedQA (PQA-L)**, 1,000 expert-labelled question–abstract pairs, is used **only to train the retrieval
  evaluator**. It is split 950 / 45 / 5 (train / val / test) with seed 42.
- **PubHealth** (`ImperialCollegeLondon/health_fact`) is the **test set**. Following Self-RAG and CRAG, only
  the binary true/false claims are kept. A fixed, label-stratified subset of 200 claims (seed 42) is used so
  every setting sees exactly the same claims (121 true, 79 false).
- **Retrieval corpus:** a health-filtered subset of English Wikipedia (`MedRAG/wikipedia`), built by
  [scripts/build_wikipedia_corpus.py](scripts/build_wikipedia_corpus.py). An article is kept if its title carries a
  medical tag such as "(disease)" or "(drug)", or if its title and opening snippet contain at least two distinct
  terms from a ~250-word health vocabulary. An earlier corpus, MedRAG Textbooks, is still supported via `--corpus textbooks`.

### 2. Retriever — [retriever.py](src/crag/retriever.py)

- **Dense retrieval with `facebook/contriever-msmarco`**, the retriever used in the CRAG paper. Claims and passages
  are embedded by the same BERT encoder (mean-pooled), and the top-k (k = 5) passages are ranked by dot product.
- Corpus embeddings are computed once, in resumable 100k-passage shards, and cached in `indexes/`.

### 3. Retrieval evaluator — [evaluator.py](src/crag/evaluator.py), trained in [pubmedqa_retrieval_evaluator.ipynb](notebooks/pubmedqa_retrieval_evaluator.ipynb)

- Architecture: a **T5-small encoder + mean pooling + linear head + tanh**. It outputs a relevance score in [-1, 1] for
  the input `"question: {q} document: {d}"`. This follows the paper's T5-based evaluator.
- Training pairs come from PubMedQA. For each question, its own abstract is a **positive (+1)** and two abstracts
  from other questions are **negatives (−1)**.
- Training settings: MSE loss, AdamW, lr 2e-5, batch size 8, 3 epochs, max length 384 tokens. The checkpoint is saved to
  `models/t5_relevance_evaluator_pubmedqa.pt`.
- Sanity check: on held-out PQA-L and 100 PQA-Artificial questions, the gold abstract should score higher than a
  random abstract.

### 4. Action trigger — [actions.py](src/crag/actions.py)

- The thresholds are applied to the per-passage scores, with `UPPER = 0.91` and `LOWER = 0.90`, tuned for the
  Wikipedia corpus. The notebook also sweeps every (upper, lower) pair over the saved forced-action predictions,
  which shows how sensitive accuracy is to the thresholds.

### 5. Knowledge refinement — [refinement.py](src/crag/refinement.py)

This implements the paper's *decompose-then-recompose* step:
1. Split each passage into **strips** of 2 sentences.
2. Score each strip against the claim with the retrieval evaluator.
3. Drop strips that score below 0, and keep the 5 highest-scoring ones.
4. Recompose them in their original reading order.

### 6. Query rewriting and web search — [web_search.py](src/crag/web_search.py)

- **Query rewriting:** the generator LLM turns the claim into 2–5 search keywords. If no LLM is available, a
  stop-word-filtering heuristic is used instead.
- **Web search:** the system queries two sources, **Tavily** (general web, which suits news-style claims) and **NCBI
  PubMed E-utilities** (peer-reviewed abstracts). Their results are interleaved, so both sources are represented,
  and each passage is clipped to 200 words.
- Results are cached in `indexes/web_search_cache.json`, which keeps runs reproducible and saves API quota. PubMed calls are
  rate-limited to stay under NCBI's 3 requests/second limit.

### 7. Generator — [generator.py](src/crag/generator.py)

- **Qwen2.5-Instruct** is the generator. The main experiments use the 3B model; 0.5B is also supported, and 7B runs
  in 4-bit. Context is truncated to 2,048 tokens.
- **Label scoring:** the label is **not** parsed from free text. After a single forward pass, the next-token logits
  for `true`/`True` and `false`/`False` are merged with log-sum-exp and renormalised with softmax. This yields a
  label plus a calibrated `P(true)` with no parsing failures.
- With `explain=True`, the generator also picks the 3 sentences that most support the predicted label, writes a
  1–2 sentence rationale, and composes a user-facing answer with a medical disclaimer. The accuracy runs use
  `explain=False`.

### 8. End-to-end pipeline — [pipeline.py](src/crag/pipeline.py)

- `run_rag` / `run_crag` return true/false labels and are used for evaluation. `ask_rag` / `ask_crag` return
  free-form cited answers and are used by the chat CLI.
- Both CRAG paths share `correct_knowledge()`, so the knowledge a user sees is exactly the knowledge that was evaluated.
- `check_crag_ready()` checks every stage, including the checkpoint, the APIs and the web search, before a long run starts.

---

## Repository structure

```
├── src/crag/                  # the CRAG implementation (one module per stage, see above)
├── scripts/
│   ├── build_wikipedia_corpus.py   # build health-Wikipedia corpus + Contriever index
│   ├── evaluate.py                 # no-retrieval vs RAG vs CRAG accuracy on PubHealth
│   └── ask.py                      # interactive RAG / CRAG medical Q&A in the terminal
├── notebooks/
│   ├── pubmedqa_retrieval_evaluator.ipynb   # train + test the T5 retrieval evaluator
│   ├── generator_evaluation_pubhealth.ipynb # main experiments, ablations, analysis, figures
│   └── retrieval_evaluator_example.ipynb    # small usage example
├── models/                    # trained evaluator checkpoint (Git LFS)
├── indexes/                   # cached Contriever embeddings (Git LFS / built locally)
└── results/                   # per-claim CSVs, knowledge logs and figures
```

---

## Setup

```bash
git lfs install
git clone https://github.com/hulemihi/42173-ANLP-CRAG.git
cd 42173-ANLP-CRAG
pip install -r requirements.txt
cp .env.example .env        # then fill in TAVILY_API_KEY and NCBI_EMAIL
```

- **Hardware:** a GPU with ≥ 16 GB VRAM runs the 3B generator in bf16. CPU or a T4 works with float32, but more slowly.
  The 0.5B model is fine for development.
- **API keys:** `TAVILY_API_KEY` is required for general web search. `NCBI_EMAIL` is required for PubMed, and
  `NCBI_API_KEY` (optional) raises the PubMed rate limit.
- The health-Wikipedia corpus and its index are too large for the repo (they are git-ignored). Build them once:

```bash
python scripts/build_wikipedia_corpus.py     # resumable; re-run after a crash
```

## Running

```bash
# Main comparison: 200 PubHealth claims, Qwen2.5-3B, health-Wikipedia corpus
python scripts/evaluate.py --crag

# Other generators / corpora
python scripts/evaluate.py --crag --model 0.5b
python scripts/evaluate.py --crag --corpus textbooks

# Check a single claim and show the retrieved passages
python scripts/evaluate.py --claim "Vitamin C cures the common cold."

# Interactive Q&A (switch between RAG and CRAG with /mode)
python scripts/ask.py --mode crag --show-retrieved
```

For the full experiments, run [notebooks/generator_evaluation_pubhealth.ipynb](notebooks/generator_evaluation_pubhealth.ipynb).
It covers the main comparison, the ablations, the threshold sweep, the knowledge-quality analysis and example
responses. Claims already in `results/` are reused (`RESUME = True`).

---

## Results

PubHealth, 200 claims (121 true / 79 false), Qwen2.5-3B-Instruct, health-Wikipedia corpus, k = 5.
Source: [results/crag_pubhealth_n200_3b_wikipedia_notebook.csv](results/crag_pubhealth_n200_3b_wikipedia_notebook.csv).

| Method | Accuracy | Macro F1 | Avg. s / claim |
|---|---|---|---|
| Always "true" (majority) | 0.605 | 0.377 | – |
| No retrieval | 0.415 | 0.341 | 0.05 |
| Standard RAG | 0.480 | 0.436 | 0.30 |
| **CRAG** | **0.705** | **0.705** | 0.82 |

**Ablations** (same claims, passages and evaluator scores; one component changed at a time,
[results/crag_ablations_…csv](results/crag_ablations_pubhealth_n200_3b_wikipedia_notebook.csv)):

| Variant | Accuracy | Macro F1 |
|---|---|---|
| CRAG (full) | 0.705 | 0.705 |
| w/o knowledge refinement | 0.725 | 0.725 |
| w/o query rewriting | 0.770 | 0.766 |
| w/o web search | 0.440 | 0.392 |
| with back-off search | 0.705 | 0.705 |
| Force *Correct* for every claim | 0.445 | 0.392 |
| Force *Ambiguous* for every claim | 0.685 | 0.685 |
| Force *Incorrect* for every claim | 0.735 | 0.734 |
| Oracle context (fact-check article) | 0.685 | 0.684 |

**Generator scale** (Qwen2.5-0.5B, [results/crag_pubhealth_n200_0.5b.csv](results/crag_pubhealth_n200_0.5b.csv)):
no retrieval 0.390, RAG 0.395, CRAG 0.490.

The notebook reports 95% bootstrap intervals, balanced accuracy, AUROC on `P(true)`, and paired McNemar tests
for every comparison.

### Key observations

- **CRAG beats standard RAG by 22.5 points** in accuracy, and is the only setting above the majority-class baseline.
  The gain holds at 0.5B (+9.5 points), so it is not tied to one model size.
- **Web search drives most of the gain.** Removing it drops accuracy to 0.440. Forcing *Incorrect* (always searching
  the web) scores 0.735, which is higher than full CRAG. PubHealth claims are mostly recent news and social-media stories,
  which a static Wikipedia corpus covers poorly.
- **The evaluator is over-confident on in-domain-looking passages.** Accuracy by CRAG action is: Incorrect 0.796
  (137 claims), Ambiguous 0.553 (38), Correct 0.44 (25). When the evaluator trusts Wikipedia, the result is worst.
  This reflects a domain shift: the evaluator was trained on PubMedQA question–abstract pairs, with random
  negatives that are easy to tell apart.
- **Query rewriting hurts here.** Searching with the raw claim (0.770) beats the LLM keywords. Short keyword
  queries lose the specific framing (who said what, when) that fact-checking pages match on.
- No retrieval scores below the majority baseline because the 3B model is biased towards "false" when it has no evidence.

## Limitations

- 200 test claims: differences of a few points fall inside the bootstrap intervals (see the notebook's paired tests).
- The evaluator is small (T5-small) and trained on 950 PubMedQA questions with random negatives. Harder negatives, such as
  passages from the same topic, would likely make its scores more discriminative.
- Web results come from live APIs. The local cache makes reruns stable, but a fresh cache may return different results.
- Thresholds were tuned on the same distribution that is evaluated; a held-out PubHealth validation split would be cleaner.

## References

- Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). *Corrective Retrieval Augmented Generation.* arXiv:2401.15884.
- Asai, A., et al. (2024). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* ICLR.
- Izacard, G., et al. (2022). *Unsupervised Dense Information Retrieval with Contrastive Learning (Contriever).* TMLR.
- Jin, Q., et al. (2019). *PubMedQA: A Dataset for Biomedical Research Question Answering.* EMNLP.
- Kotonya, N., & Toni, F. (2020). *Explainable Automated Fact-Checking for Public Health Claims (PubHealth).* EMNLP.
- Xiong, G., et al. (2024). *Benchmarking Retrieval-Augmented Generation for Medicine (MedRAG).* ACL Findings.
- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.

## License

See [LICENSE](LICENSE).
