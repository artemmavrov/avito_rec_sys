# Candidate generation for Avito service search

For every search query, return up to 50 `item_id`s from a corpus of 189k service ads. Metric: Recall@50.

The solution is a cascade. Recall matters most at this stage, so every stage keeps a wide pool and the cut to 50 happens once, at the end.

```
query (text + filters + location + category)
  |-- BM25 over lemmas, 3 fields (title / filtered params / description)  --.
  |-- bge-m3 dense                                                         |-- weighted RRF per ranking (local, global)
  |-- bge-m3 sparse (lexical weights)                                     --'
  |        + list of global candidates within 100 km of the query location
  |          (location centroid from corpus items, else from train clicks)
  v
RRF merge of the three lists -> pool of 300 (+ log candidates: q->item, title bridge from train)
  v
feature bank (35 lexical/geo/log features + dense / sparse / ColBERT scores, their in-pool ranks)
  + bge-reranker-v2-m3 cross-encoder logit
  v
CatBoost YetiRank (Recall@50 early stopping)
  v
top 50 by ranker score (a local/global slot quota is also evaluated; the better one on validation is used)
```

Both neural models are fine-tuned on the training pairs (frozen embedding table + 6 lowest layers).

## Open-source components used

| Component | Use | License |
|---|---|---|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (pinned revision in `configs/models.yaml`) | dense + sparse + ColBERT bi-encoder, fine-tuned locally | MIT |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) (pinned) | cross-encoder, fine-tuned locally | MIT |
| [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) | bge-m3 inference wrapper and its unified dense+sparse+ColBERT fine-tuning trainer | MIT |
| PyTorch, Hugging Face `transformers`, `accelerate` | model code, training loop | BSD / Apache-2.0 |
| CatBoost | ranking model (YetiRank) | Apache-2.0 |
| pymorphy3 (+ `pymorphy3-dicts-ru`) | Russian lemmatization for lexical features | MIT |
| polars, pandas, pyarrow, NumPy, SciPy | data handling, sparse BM25 | BSD / Apache-2.0 |

The multi-field BM25, RRF, slot allocation, feature bank, negative mining and the validation protocol are written for this task. No external API is called at any point; after the model weights are downloaded once, everything runs offline.

## Layout

```
configs/            pipeline.yaml (all hyper-parameters), models.yaml (model ids + revisions), smoke_test.yaml
avito_rec_sys/
  data/             loading, query normalization, item-params parser, train cleaning, leak-free split, lemmatization
  features/         multi-field BM25, tables, log signals, the feature bank
  retrieval/        bge-m3 wrapper, dense/sparse tours, RRF, ColBERT narrowing, slots, cross-encoder scorer
  training/         bi-encoder + cross-encoder fine-tuning, hard-negative mining, CatBoost, layer freezing
  inference/        stage pipelines, slot selection, answer writer + validator
  eval/             Recall@K, strata, ceiling curves
scripts/            s01 ... s11, one per pipeline stage (see below)
tests/              unit tests + GPU smoke tests
```

## Reproduce

```bash
pip install -r requirements.txt          # torch: use the CUDA wheel matching your driver
pytest tests -m "not gpu"                # CPU unit tests

# CPU stages (validated locally): splits, cleaning, BM25 baseline, lexical CatBoost, baseline answer
python scripts/s01_split_and_normalize.py
python scripts/s02_clean_lemmatize_params.py
python scripts/s03_bm25f_baseline.py
python scripts/s04_catboost_lexical.py

# GPU stages (one 24 GB GPU; see reports/ for the time budget and measured timings)
python scripts/s05_zeroshot_biencoder.py [--cpu]   # zero-shot encode + ceiling curve + RRF weights
python scripts/s06_mine_hard_negatives.py
python scripts/s07_train_biencoder.py              # 2 epochs, negative refresh between them (~2.2 h on a 3090)
python scripts/s08_reencode_mine_pools.py [--store-only]   # store_ft, then cross-encoder training groups
python scripts/s09_zeroshot_reranker_eval.py [--max-queries 1200]   # gate: is cross-encoder fine-tuning worth it?
python scripts/s10_train_reranker.py               # only if the gate says so (~1.3 h)
# s11 builds feature tables per stage (cached in work/feats_*.parquet); the three stages are independent,
# so they can be run as concurrent processes to overlap CPU phases with cross-encoder scoring:
python scripts/s11_final_inference.py --features-only ranker &
python scripts/s11_final_inference.py --features-only val && python scripts/s11_final_inference.py --features-only test
python scripts/s11_final_inference.py [--zero-shot-ce]   # trains CatBoost, writes answer.csv and validates it
```

`AVITO_WORKERS` (default: cores - 2, at most 14) sets the number of forked processes for the ColBERT MaxSim step.
On a 24 GB card the bi-encoder fine-tuning at 256 groups per batch relies on two memory patches to FlagEmbedding's M3
trainer (`sparse_embedding_lowmem`, `colbert_score_chunked` in `training/biencoder_train.py`); both are covered by tests.

Data is read from `../data`, intermediate artifacts go to `../work` (paths in `configs/pipeline.yaml`).
All training stages are resumable from their last checkpoint.

`pytest tests -m gpu` runs small real-model smoke tests (a few training steps, checkpoint save/reload, an end-to-end pipeline slice); it needs a CUDA GPU with ~4 GB.

## Validation protocol

The hold-outs are built to look like the benchmark (see `avito_rec_sys/data/split.py`): split on the *normalized* query text (word-sorted, so spelling variants cannot leak), one random `search_*` tuple per held-out text, 38.5% of hold-out queries keep their text in train (as in the benchmark) and the rest do not, held-out positives are removed from the training part, and the distractor pool is the real benchmark corpus. Two disjoint hold-outs are used: `val` for measurement and `ranker` for training CatBoost, so the ranker never sees encoder-memorised pairs.
