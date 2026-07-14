# FAISS-Based Retrieval Optimization for Low-Resource Arabic QA

Benchmarking FAISS index types (Flat, IVF-PQ, HNSW) against a BM25 baseline for Arabic question-answering retrieval, evaluated on accuracy, latency, and index size.

## Key finding

BM25 beats a general-purpose multilingual dense retriever by 10–15 recall points at every k, on every corpus tested — the opposite of what's usually assumed about Arabic retrieval. Within FAISS itself, HNSW is effectively lossless versus exact search, while IVF-PQ trades a small accuracy loss for the smallest index footprint.

| Corpus | Method | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---|---|---|---|---|---|
| ARCD (N=465) | Flat | 30.6 | 51.0 | 58.3 | 65.1 | .402 |
| ARCD (N=465) | **BM25** | **45.9** | **70.5** | **75.9** | **79.7** | **.569** |
| TyDi-AR (N=842) | Flat | 44.5 | 62.1 | 68.9 | 73.1 | .522 |
| TyDi-AR (N=842) | **BM25** | **54.8** | **72.2** | **78.1** | **82.0** | **.629** |
| Combined (N=1307) | Flat | 32.3 | 50.4 | 56.4 | 61.9 | .404 |
| Combined (N=1307) | **BM25** | **44.0** | **66.5** | **72.4** | **77.3** | **.540** |

Full comparison across all four methods (Flat, IVF-PQ, HNSW, BM25) and the accuracy-latency sweeps are in [`results/results.json`](results/results.json).

Results were independently reproduced end-to-end on two architecturally different machines (x86/aarch64 Linux and Apple Silicon macOS), with identical recall/MRR values on both.

## Repo structure

```
code/       Reproduction scripts
data/       Small, redistributable dataset files (ARCD + TyDi QA Arabic subset)
results/    results.json — the numbers behind every table in the paper
```

## Reproducing the results

**1. Install dependencies**

```bash
python3 -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**2. Get the embedding model**

Download [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) into `model_minilm/` at the repo root:

```bash
huggingface-cli download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --local-dir model_minilm
```

**3. Run**

```bash
python3 code/run_experiment.py
```

This encodes both corpora, builds all four index types, runs the IVF-PQ/HNSW parameter sweeps and the BM25 baseline, and writes `results/results.json`. Runtime is a few minutes on a laptop CPU.

> **Apple Silicon note:** PyTorch and `faiss-cpu` each bundle their own OpenMP runtime, which can crash on macOS when both run multi-threaded in the same process. `run_experiment.py` already pins both libraries to a single thread to avoid this — no action needed.

## Datasets

- [ARCD](https://github.com/husseinmozannar/SOQAL) — Arabic Reading Comprehension Dataset, 465 passages / 1,395 questions.
- [TyDi QA](https://huggingface.co/datasets/google-research-datasets/tydiqa) — gold-passage task, Arabic subset, 842 passages / 921 questions.

Both are included pre-processed in `data/`. Arabic text is normalized before encoding (diacritics and tatweel stripped, alef/ya/ta-marbuta variants unified) — see `normalize_arabic()` in `code/run_experiment.py`.



## License

Code is released under the [MIT License](LICENSE). ARCD and TyDi QA retain their original dataset licenses.

