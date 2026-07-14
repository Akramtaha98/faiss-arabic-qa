"""Build FAISS indexes from cached embeddings and evaluate. Also runs BM25.
Reads: texts/*.json, results/emb_*.npy
Writes: results/results.json
"""
import json
import os
import re
import statistics as stats
import time

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

OUT_DIR = "results"

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


def build_flat(dim):
    return faiss.IndexFlatIP(dim)


def build_ivfpq(dim, nlist):
    m = 48 if dim % 48 == 0 else (32 if dim % 32 == 0 else 16)
    quantizer = faiss.IndexFlatIP(dim)
    return faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)


def build_hnsw(dim, M=32, ef_construction=40):
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    return index


_tmp_counter = [0]


def index_size_bytes(index):
    _tmp_counter[0] += 1
    tmp_path = os.path.join(OUT_DIR, f"_tmp_index_{_tmp_counter[0]}.faiss")
    faiss.write_index(index, tmp_path)
    size = os.path.getsize(tmp_path)
    return size


def evaluate_index(index, query_embs, gold_idxs, k_max=20, n_warmup=5):
    n = len(query_embs)
    ranks, latencies = [], []
    for i in range(n):
        q = query_embs[i : i + 1]
        t0 = time.perf_counter()
        _, I = index.search(q, k_max)
        t1 = time.perf_counter()
        if i >= n_warmup:
            latencies.append((t1 - t0) * 1000.0)
        retrieved = I[0].tolist()
        gold = gold_idxs[i]
        ranks.append(retrieved.index(gold) + 1 if gold in retrieved else None)

    def recall_at(k):
        return sum(1 for r in ranks if r is not None and r <= k) / n

    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    return {
        "recall@1": recall_at(1), "recall@5": recall_at(5),
        "recall@10": recall_at(10), "recall@20": recall_at(20),
        "mrr": mrr,
        "latency_ms_mean": stats.mean(latencies) if latencies else float("nan"),
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else float("nan"),
        "n_queries": n,
    }


def evaluate_bm25(passages, questions, gold_idxs, k_max=20):
    tok_passages = [normalize_arabic(p).split() for p in passages]
    bm25 = BM25Okapi(tok_passages)
    ranks, latencies = [], []
    for i, q in enumerate(questions):
        q_tok = normalize_arabic(q).split()
        t0 = time.perf_counter()
        scores = bm25.get_scores(q_tok)
        top_idx = np.argsort(scores)[::-1][:k_max].tolist()
        t1 = time.perf_counter()
        if i >= 5:
            latencies.append((t1 - t0) * 1000.0)
        gold = gold_idxs[i]
        ranks.append(top_idx.index(gold) + 1 if gold in top_idx else None)
    n = len(ranks)

    def recall_at(k):
        return sum(1 for r in ranks if r is not None and r <= k) / n

    mrr = sum((1.0 / r) if r is not None else 0.0 for r in ranks) / n
    return {
        "recall@1": recall_at(1), "recall@5": recall_at(5),
        "recall@10": recall_at(10), "recall@20": recall_at(20),
        "mrr": mrr,
        "latency_ms_mean": stats.mean(latencies) if latencies else float("nan"),
        "latency_ms_p95": float(np.percentile(latencies, 95)) if latencies else float("nan"),
        "n_queries": n,
    }


def run_dataset(name, passage_embs, query_embs, gold_idxs, passages_text, questions_text,
                 nlist, nprobe_grid, efsearch_grid):
    dim = passage_embs.shape[1]
    results = {"dataset": name, "n_passages": len(passage_embs), "n_questions": len(query_embs)}

    idx = build_flat(dim)
    t0 = time.perf_counter()
    idx.add(passage_embs)
    bt = time.perf_counter() - t0
    m = evaluate_index(idx, query_embs, gold_idxs)
    m["build_time_s"] = bt
    m["index_size_bytes"] = index_size_bytes(idx)
    results["flat"] = m

    nlist_eff = min(nlist, max(4, len(passage_embs) // 5))
    idx = build_ivfpq(dim, nlist_eff)
    t0 = time.perf_counter()
    idx.train(passage_embs)
    idx.add(passage_embs)
    bt = time.perf_counter() - t0
    curve = []
    for nprobe in nprobe_grid:
        idx.nprobe = min(nprobe, nlist_eff)
        m = evaluate_index(idx, query_embs, gold_idxs)
        m.update({"nprobe": idx.nprobe, "nlist": nlist_eff, "build_time_s": bt,
                   "index_size_bytes": index_size_bytes(idx)})
        curve.append(m)
    results["ivfpq_sweep"] = curve

    idx = build_hnsw(dim, M=32, ef_construction=40)
    t0 = time.perf_counter()
    idx.add(passage_embs)
    bt = time.perf_counter() - t0
    curve = []
    for ef in efsearch_grid:
        idx.hnsw.efSearch = ef
        m = evaluate_index(idx, query_embs, gold_idxs)
        m.update({"efSearch": ef, "M": 32, "build_time_s": bt,
                   "index_size_bytes": index_size_bytes(idx)})
        curve.append(m)
    results["hnsw_sweep"] = curve

    results["bm25"] = evaluate_bm25(passages_text, questions_text, gold_idxs)
    return results


def load(name):
    p = np.load(f"results/emb_{name}_v2.npy")
    return p


if __name__ == "__main__":
    all_results = {}

    arcd_p = load("arcd_passages")
    arcd_q = load("arcd_questions")
    arcd_gold = json.load(open("texts/arcd_gold.json"))
    arcd_ptext = json.load(open("texts/arcd_passages.json", encoding="utf-8"))
    arcd_qtext = json.load(open("texts/arcd_questions.json", encoding="utf-8"))
    all_results["ARCD"] = run_dataset(
        "ARCD", arcd_p, arcd_q, arcd_gold, arcd_ptext, arcd_qtext,
        nlist=32, nprobe_grid=[1, 2, 4, 8, 16, 32], efsearch_grid=[8, 16, 32, 64, 128],
    )
    print("ARCD done")

    tydi_p = load("tydi_passages")
    tydi_q = load("tydi_questions")
    tydi_gold = json.load(open("texts/tydi_gold.json"))
    tydi_ptext = json.load(open("texts/tydi_passages.json", encoding="utf-8"))
    tydi_qtext = json.load(open("texts/tydi_questions.json", encoding="utf-8"))
    all_results["TyDiQA-Arabic"] = run_dataset(
        "TyDiQA-Arabic", tydi_p, tydi_q, tydi_gold, tydi_ptext, tydi_qtext,
        nlist=48, nprobe_grid=[1, 2, 4, 8, 16, 32, 48], efsearch_grid=[8, 16, 32, 64, 128],
    )
    print("TyDiQA-Arabic done")

    combined_p = np.concatenate([arcd_p, tydi_p], axis=0)
    combined_ptext = arcd_ptext + tydi_ptext
    offset = len(arcd_ptext)
    combined_q = np.concatenate([arcd_q, tydi_q], axis=0)
    combined_qtext = arcd_qtext + tydi_qtext
    combined_gold = arcd_gold + [g + offset for g in tydi_gold]
    all_results["Combined"] = run_dataset(
        "Combined", combined_p, combined_q, combined_gold, combined_ptext, combined_qtext,
        nlist=64, nprobe_grid=[1, 2, 4, 8, 16, 32, 64], efsearch_grid=[8, 16, 32, 64, 128],
    )
    print("Combined done")

    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print("Saved results/results.json")
