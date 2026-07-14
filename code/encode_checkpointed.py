"""Checkpointed encoder: run repeatedly (each invocation has a wall-clock
budget) until it prints DONE. Saves progress to <out>.npy + <out>.progress
after every batch so a call that gets cut off resumes cleanly next time.
"""
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "./model_minilm"
TIME_BUDGET_S = float(os.environ.get("TIME_BUDGET_S", "35"))

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


def mean_pool(out, mask):
    te = out[0]
    m = mask.unsqueeze(-1).expand(te.size()).float()
    return torch.sum(te * m, 1) / torch.clamp(m.sum(1), min=1e-9)


def main():
    texts_path = sys.argv[1]  # json file: list of strings
    out_prefix = sys.argv[2]  # e.g. results/emb_arcd_passages
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    max_length = int(sys.argv[4]) if len(sys.argv) > 4 else 256

    texts = json.load(open(texts_path, encoding="utf-8"))
    n = len(texts)
    emb_path = out_prefix + ".npy"
    progress_path = out_prefix + ".progress"

    start = 0
    embs = []
    if os.path.exists(progress_path) and os.path.exists(emb_path):
        start = int(open(progress_path).read().strip())
        embs = [np.load(emb_path)]
        print(f"resuming from {start}/{n}")

    if start >= n:
        print("DONE")
        return

    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH)
    model.eval()

    t_start = time.perf_counter()
    i = start
    with torch.no_grad():
        while i < n:
            if time.perf_counter() - t_start > TIME_BUDGET_S:
                break
            batch = [normalize_arabic(t) for t in texts[i : i + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            out = model(**enc)
            pooled = mean_pool(out, enc["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            embs.append(pooled.numpy().astype("float32"))
            i += len(batch)

    all_embs = np.concatenate(embs, axis=0)
    np.save(emb_path, all_embs)
    with open(progress_path, "w") as f:
        f.write(str(i))

    print(f"progress: {i}/{n}")
    if i >= n:
        print("DONE")


if __name__ == "__main__":
    main()
