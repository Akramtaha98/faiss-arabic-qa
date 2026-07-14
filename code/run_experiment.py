"""
FAISS-Based Retrieval Optimization for Low-Resource Arabic QA
Reproducible experiment script (Section IV protocol).

Usage:
    python3 run_experiment.py

Requires: torch, transformers, faiss-cpu, rank_bm25, numpy, pandas.
Downloads: none at runtime (model + data are loaded from local paths below;
see fetch instructions in the paper's code-availability note / README).

Output: results/results.json with recall@k, MRR, latency (mean/p95), and
index size for every (dataset x index configuration) cell, plus the IVF/HNSW
parameter sweeps used to build the accuracy-latency curves in Section V.
"""
import os

# macOS note: PyTorch and faiss-cpu each bundle their own OpenMP runtime
# (libomp.dylib). Loading both in one process aborts with "OMP: Error #15"
# unless this is set before either library is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import re
import time
import statistics as stats

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss
from rank_bm25 import BM25Okapi

# macOS ARM64 note: faiss-cpu's multi-threaded k-means (used by IndexIVFPQ.train)
# can segfault inside libomp when a second OpenMP-using library (PyTorch) is also
# loaded in-process, even with KMP_DUPLICATE_LIB_OK=TRUE set. FAISS's own clustering
# workload here is tiny (a few hundred to ~1,300 points), so single-threaded FAISS
# costs essentially nothing and avoids the crash entirely.
faiss.omp_set_num_threads(1)
torch.set_num_threads(1)

MODEL_PATH = os.environ.get("MODEL_PATH", "./model_minilm")
DATA_DIR = os.environ.get("DATA_DIR", "./data")
OUT_DIR = os.environ.get("OUT_DIR", "./results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Arabic text normalization (diacritics / alef / ya / ta-marbuta), per §III-C
# ---------------------------------------------------------------------------
_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]")
_TATWEEL = re.compile(r"ـ")


def normalize_arabic(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Dataset loaders -> (passages: List[str], questions: List[{question, gold_idx}])
# ---------------------------------------------------------------------------
def load_arcd(path):
    d = json.load(open(path, encoding="utf-8"))
    passages, questions = [], []
    for article in d["data"]:
        for para in article["paragraphs"]:
            pid = len(passages)
            passages.append(para["context"])
            for qa in para["qas"]:
                q = qa["question"].strip()
                if q:
                    questions.append({"question": q, "gold_idx": pid})
    return passages, questions


def load_tydi_arabic(path):
    df = pd.read_parquet(path)
    df = df[df["id"].str.startswith("arabic")]
    ctx_to_idx, passages, questions = {}, [], []
    for _, row in df.iterrows():
        ctx = row["context"]
        if ctx not in ctx_to_idx:
            ctx_to_idx[ctx] = len(passages)
            passages.append(ctx)
        q = str(row["question"]).strip()
        if q:
            questions.append({"question": q, "gold_idx": ctx_to_idx[ctx]})
    return passages, questions


# ---------------------------------------------------------------------------
# Embedding model: paraphrase-multilingual-MiniLM-L12-v2, mean pooling, L2 norm
# ---------------------------------------------------------------------------
print("Loading embedding model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)
model.eval()


def mean_pool(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


@torch.no_grad()
def encode(texts, batch_size=32, max_length=256):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = [normalize_arabic(t) for t in texts[i : i + batch_size]]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        out = model(**enc)
        pooled = mean_pool(out, enc["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        embs.append(pooled.numpy())
    return np.concatenate(embs, axis=0).astype("float32")


# ---------------------------------------------------------------------------
# FAISS index builders
# ---------------------------------------------------------------------------
def build_flat(dim):
    return faiss.IndexFlatIP(dim)


def build_ivfpq(dim, n_train, nlist):
    m = 48 if dim % 48 == 0 else (32 if dim % 32 == 0 else 16)
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
    return index


def build_hnsw(dim, M=32, ef_construction=40):
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    return index


def index_size_bytes(index):
    tmp_path = os.path.join(OUT_DIR, "_tmp_index.faiss")
    faiss.write_index(index, tmp_path)
    size = os.path.getsize(tmp_path)
    os.remove(tmp_path)
    return size


# ---------------------------------------------------------------------------
# Retrieval evaluation: recall@k, MRR, latency
# ---------------------------------------------------------------------------
def evaluate_index(index, query_embs, gold_idxs, k_max=20, n_warmup=5):
    n = len(query_embs)
    ranks = []
    latencies = []
    for i in range(n):
        q = query_embs[i : i + 1]
        t0 = time.perf_counter()
        _, I = index.search(q, k_max)
        t1 = time.perf_counter()
        if i >= n_warmup:
            latencies.append((t1 - t0) * 1000.0)  # ms
        retrieved = I[0].tolist()
        gold = gold_idxs[i]
        if gold in retrieved:
            ranks.append(retrieved.index(gold) + 1)
        else:
            ranks.append(None)

    def recall_at(k):
        hits = sum(1 for r in ranks if r is not None and r <= k)
        return hits / n

    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    lat_mean = stats.mean(latencies) if latencies else float("nan")
    lat_p95 = np.percentile(latencies, 95) if latencies else float("nan")
    return {
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "recall@20": recall_at(20),
        "mrr": mrr,
        "latency_ms_mean": lat_mean,
        "latency_ms_p95": lat_p95,
        "n_queries": n,
    }


def evaluate_bm25(passages_norm_tok, questions, gold_idxs, k_max=20):
    bm25 = BM25Okapi(passages_norm_tok)
    ranks = []
    latencies = []
    for i, q in enumerate(questions):
        q_tok = normalize_arabic(q).split()
        t0 = time.perf_counter()
        scores = bm25.get_scores(q_tok)
        top_idx = np.argsort(scores)[::-1][:k_max]
        t1 = time.perf_counter()
        if i >= 5:
            latencies.append((t1 - t0) * 1000.0)
        gold = gold_idxs[i]
        top_idx = top_idx.tolist()
        ranks.append(top_idx.index(gold) + 1 if gold in top_idx else None)

    def recall_at(k):
        hits = sum(1 for r in ranks if r is not None and r <= k)
        return hits / len(ranks)

    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / len(ranks)
    return {
        "recall@1": recall_at(1),
        "recall@5": recall_at(5),
        "recall@10": recall_at(10),
        "recall@20": recall_at(20),
        "mrr": mrr,
        "latency_ms_mean": stats.mean(latencies) if latencies else float("nan"),
        "latency_ms_p95": np.percentile(latencies, 95) if latencies else float("nan"),
        "n_queries": len(ranks),
    }


# ---------------------------------------------------------------------------
# Run one dataset end to end
# ---------------------------------------------------------------------------
def run_dataset(name, passages, questions, ivf_nlist, nprobe_grid, efsearch_grid):
    print(f"\n=== {name}: {len(passages)} passages, {len(questions)} questions ===")
    t0 = time.perf_counter()
    passage_embs = encode(passages)
    query_embs = encode([q["question"] for q in questions])
    gold_idxs = [q["gold_idx"] for q in questions]
    print(f"  encoding done in {time.perf_counter()-t0:.1f}s, dim={passage_embs.shape[1]}")
    dim = passage_embs.shape[1]

    results = {"dataset": name, "n_passages": len(passages), "n_questions": len(questions)}

    # --- Flat (exact) ---
    idx = build_flat(dim)
    t0 = time.perf_counter()
    idx.add(passage_embs)
    build_time = time.perf_counter() - t0
    metrics = evaluate_index(idx, query_embs, gold_idxs)
    metrics["build_time_s"] = build_time
    metrics["index_size_bytes"] = index_size_bytes(idx)
    results["flat"] = metrics
    print("  flat:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    # --- IVF-PQ sweep over nprobe ---
    nlist = min(ivf_nlist, max(4, len(passages) // 5))
    idx = build_ivfpq(dim, len(passages), nlist)
    t0 = time.perf_counter()
    idx.train(passage_embs)
    idx.add(passage_embs)
    build_time = time.perf_counter() - t0
    ivf_curve = []
    for nprobe in nprobe_grid:
        idx.nprobe = min(nprobe, nlist)
        m = evaluate_index(idx, query_embs, gold_idxs)
        m["nprobe"] = idx.nprobe
        m["nlist"] = nlist
        m["build_time_s"] = build_time
        m["index_size_bytes"] = index_size_bytes(idx)
        ivf_curve.append(m)
        print(f"  ivfpq nprobe={idx.nprobe}:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k not in ("nprobe", "nlist")})
    results["ivfpq_sweep"] = ivf_curve

    # --- HNSW sweep over efSearch ---
    idx = build_hnsw(dim, M=32, ef_construction=40)
    t0 = time.perf_counter()
    idx.add(passage_embs)
    build_time = time.perf_counter() - t0
    hnsw_curve = []
    for ef in efsearch_grid:
        idx.hnsw.efSearch = ef
        m = evaluate_index(idx, query_embs, gold_idxs)
        m["efSearch"] = ef
        m["M"] = 32
        m["build_time_s"] = build_time
        m["index_size_bytes"] = index_size_bytes(idx)
        hnsw_curve.append(m)
        print(f"  hnsw efSearch={ef}:", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items() if k not in ("efSearch", "M")})
    results["hnsw_sweep"] = hnsw_curve

    # --- BM25 baseline ---
    passages_norm_tok = [normalize_arabic(p).split() for p in passages]
    bm25_metrics = evaluate_bm25(passages_norm_tok, [q["question"] for q in questions], gold_idxs)
    results["bm25"] = bm25_metrics
    print("  bm25:", {k: round(v, 4) if isinstance(v, float) else v for k, v in bm25_metrics.items()})

    return results


if __name__ == "__main__":
    all_results = {}

    arcd_passages, arcd_questions = load_arcd(os.path.join(DATA_DIR, "arcd.json"))
    all_results["ARCD"] = run_dataset(
        "ARCD", arcd_passages, arcd_questions,
        ivf_nlist=32, nprobe_grid=[1, 2, 4, 8, 16, 32], efsearch_grid=[8, 16, 32, 64, 128],
    )

    tydi_passages, tydi_questions = load_tydi_arabic(os.path.join(DATA_DIR, "tydiqa_arabic_val.parquet"))
    all_results["TyDiQA-Arabic"] = run_dataset(
        "TyDiQA-Arabic", tydi_passages, tydi_questions,
        ivf_nlist=48, nprobe_grid=[1, 2, 4, 8, 16, 32, 48], efsearch_grid=[8, 16, 32, 64, 128],
    )

    # Combined corpus (ARCD + TyDi passages pooled) to approximate a more
    # realistic low-resource deployment scale; queries evaluated against the
    # pooled corpus with gold indices offset for TyDi passages.
    combined_passages = arcd_passages + tydi_passages
    offset = len(arcd_passages)
    combined_questions = (
        [{"question": q["question"], "gold_idx": q["gold_idx"]} for q in arcd_questions]
        + [{"question": q["question"], "gold_idx": q["gold_idx"] + offset} for q in tydi_questions]
    )
    all_results["Combined"] = run_dataset(
        "Combined", combined_passages, combined_questions,
        ivf_nlist=64, nprobe_grid=[1, 2, 4, 8, 16, 32, 64], efsearch_grid=[8, 16, 32, 64, 128],
    )

    out_path = os.path.join(OUT_DIR, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {out_path}")
