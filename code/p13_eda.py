"""
p13_eda.py - Exploratory Data Analysis for the oncology distress/response corpus.

Generates publication-style figures (saved as PNG) and a short text summary describing the corpus:
class balance (response + distress), message length, the cross-tabulation of the two label axes,
split sizes, and the generation attributes (role/stage/cancer_type/tone/channel/etc).

Reads the tier files and the splits; writes figures to visuals/ and a summary to
results/eda_summary.txt.

USAGE (from project root):
    py p13_eda.py
"""

import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESPONSE = ["anxiety", "sadness", "anger", "hope", "guilt", "denial", "acceptance"]
DISTRESS = ["low", "medium", "high"]

SPLITS = "data/gen_v6_low_medium/splits"
TIERS = "data/gen_v6_low_medium/tiers"
MODEL_READY = TIERS + "/dataset_model_ready.jsonl"
ALL_FILE = TIERS + "/dataset_all.jsonl"
SPLIT_FILES = {"train_A": SPLITS + "/train_A.jsonl", "train_B": SPLITS + "/train_B.jsonl",
               "train_C": SPLITS + "/train_C.jsonl", "validation": SPLITS + "/validation.jsonl",
               "test": SPLITS + "/test.jsonl"}

VIS = Path("visuals")
VIS.mkdir(parents=True, exist_ok=True)
Path("results").mkdir(parents=True, exist_ok=True)

# consistent colors
C_MAIN = "#2c7fb8"
C_ACCENT = "#de2d26"
PALETTE = ["#2c7fb8", "#7fcdbb", "#de2d26", "#fdae6b", "#756bb1", "#31a354", "#636363"]


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bar(ax, labels, counts, title, color=C_MAIN, rotate=0):
    ax.bar(range(len(labels)), counts, color=color, edgecolor="black", linewidth=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i, c in enumerate(counts):
        ax.text(i, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    summary = []
    summary.append("EDA SUMMARY - oncology distress/response corpus")
    summary.append("=" * 55)

    model_ready = read_jsonl(MODEL_READY)
    summary.append("model-ready records: {}".format(len(model_ready)))

    # ---- Figure 1: response + distress class balance (model-ready) ----
    resp_counts = Counter(r["final_response"] for r in model_ready)
    dist_counts = Counter(r["final_distress"] for r in model_ready)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bar(axes[0], RESPONSE, [resp_counts.get(l, 0) for l in RESPONSE],
        "Response class balance (model-ready, n={})".format(len(model_ready)), C_MAIN, rotate=30)
    bar(axes[1], DISTRESS, [dist_counts.get(l, 0) for l in DISTRESS],
        "Distress level balance (model-ready)", C_ACCENT)
    fig.tight_layout()
    fig.savefig(VIS / "eda_class_balance.png", dpi=150)
    plt.close(fig)
    summary.append("\nresponse balance: {}".format({l: resp_counts.get(l, 0) for l in RESPONSE}))
    summary.append("distress balance: {}".format({l: dist_counts.get(l, 0) for l in DISTRESS}))

    # ---- Figure 2: message length distribution (words) ----
    lengths = [len(r["text"].split()) for r in model_ready]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(lengths, bins=30, color=C_MAIN, edgecolor="black", linewidth=0.5)
    ax.axvline(np.median(lengths), color=C_ACCENT, linestyle="--",
               label="median {}".format(int(np.median(lengths))))
    ax.set_xlabel("message length (words)")
    ax.set_ylabel("count")
    ax.set_title("Message length distribution (model-ready)", fontsize=11, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(VIS / "eda_length.png", dpi=150)
    plt.close(fig)
    summary.append("\nmessage length (words): min {} median {} mean {:.1f} max {}".format(
        min(lengths), int(np.median(lengths)), float(np.mean(lengths)), max(lengths)))

    # ---- Figure 3: response x distress heatmap (model-ready) ----
    mat = np.zeros((len(RESPONSE), len(DISTRESS)), dtype=int)
    for r in model_ready:
        ri = RESPONSE.index(r["final_response"]) if r["final_response"] in RESPONSE else None
        di = DISTRESS.index(r["final_distress"]) if r["final_distress"] in DISTRESS else None
        if ri is not None and di is not None:
            mat[ri][di] += 1
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(DISTRESS))); ax.set_xticklabels(DISTRESS)
    ax.set_yticks(range(len(RESPONSE))); ax.set_yticklabels(RESPONSE)
    ax.set_xlabel("distress"); ax.set_ylabel("response")
    ax.set_title("Response x Distress co-occurrence (model-ready)", fontsize=11, fontweight="bold")
    for i in range(len(RESPONSE)):
        for j in range(len(DISTRESS)):
            ax.text(j, i, str(mat[i][j]), ha="center", va="center",
                    color="white" if mat[i][j] > mat.max() * 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(VIS / "eda_response_distress_heatmap.png", dpi=150)
    plt.close(fig)

    # ---- Figure 4: split sizes + per-split response balance (stacked) ----
    split_resp = {}
    sizes = {}
    for name, path in SPLIT_FILES.items():
        rows = read_jsonl(path)
        sizes[name] = len(rows)
        split_resp[name] = Counter(r["final_response"] for r in rows)
    order = ["train_A", "train_B", "train_C", "validation", "test"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    bar(axes[0], order, [sizes[s] for s in order], "Split sizes", C_MAIN, rotate=20)
    # stacked response composition per split
    bottoms = np.zeros(len(order))
    for ci, cls in enumerate(RESPONSE):
        vals = [split_resp[s].get(cls, 0) for s in order]
        axes[1].bar(range(len(order)), vals, bottom=bottoms, label=cls,
                    color=PALETTE[ci % len(PALETTE)], edgecolor="white", linewidth=0.3)
        bottoms += np.array(vals)
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(order, rotation=20, ha="right", fontsize=9)
    axes[1].set_title("Response composition per split", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(VIS / "eda_splits.png", dpi=150)
    plt.close(fig)
    summary.append("\nsplit sizes: {}".format({s: sizes[s] for s in order}))

    # ---- Figure 5: generation attributes (from dataset_all, the full audited corpus) ----
    all_rows = read_jsonl(ALL_FILE)
    attr_keys = ["role", "stage", "cancer_type", "tone", "channel", "age_group", "length"]
    present = [k for k in attr_keys if any("attributes" in r and k in r.get("attributes", {}) for r in all_rows)]
    nshow = present[:6]
    if nshow:
        ncol = 3
        nrow = int(np.ceil(len(nshow) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4.2 * nrow))
        axes = np.array(axes).reshape(-1)
        for idx, key in enumerate(nshow):
            c = Counter(r["attributes"].get(key, "NA") for r in all_rows if "attributes" in r)
            items = c.most_common(8)
            labels = [str(k) for k, _ in items]
            counts = [v for _, v in items]
            bar(axes[idx], labels, counts, "attribute: {}".format(key),
                PALETTE[idx % len(PALETTE)], rotate=30)
        for j in range(len(nshow), len(axes)):
            axes[j].axis("off")
        fig.suptitle("Generation attributes (full audited corpus, n={})".format(len(all_rows)),
                     fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(VIS / "eda_attributes.png", dpi=150)
        plt.close(fig)
        summary.append("\nattributes summarized: {}".format(nshow))

    # ---- write summary ----
    with open("results/eda_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print("EDA done. Figures in visuals/:")
    for p in sorted(VIS.glob("eda_*.png")):
        print("  {}".format(p))
    print("Summary: results/eda_summary.txt")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
