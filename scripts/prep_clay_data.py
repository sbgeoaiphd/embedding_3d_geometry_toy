#!/usr/bin/env python3
"""
Prepare Clay land-cover data for the 3D embedding viewer.

Reads concept-lab parquets + labels + concept models and outputs:
  data/clay_embeddings.json  – [{id, class, embedding}, ...]
  data/clay_class_ws.json    – {concept_name: [1024 floats], ...}
"""

import json
import os
import sys
import numpy as np

CONCEPT_LAB = "/Users/lgnd/Documents/repos/concept-lab/data/general/.concept_lab"
PARQUET_DIR = "/Users/lgnd/Documents/repos/concept-lab/data/general"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# The 6 display categories (chip class labels)
WANTED_CATEGORIES = {
    "agriculture",
    "forest",
    "desert",
    "CBD",
    "suburbs",
    "rural_homes",
}

# Nice display names
DISPLAY_NAMES = {
    "agriculture": "Agriculture",
    "forest": "Forest",
    "desert": "Desert",
    "CBD": "CBD",
    "suburbs": "Suburbs",
    "rural_homes": "Rural Housing",
}


def load_all_parquets(parquet_dir):
    """Load all parquets and deduplicate by CHIPS_ID, returning {chips_id: embedding}."""
    import pyarrow.parquet as pq

    id_to_emb = {}
    for fname in sorted(os.listdir(parquet_dir)):
        if not fname.endswith(".parquet"):
            continue
        path = os.path.join(parquet_dir, fname)
        schema = pq.read_schema(path)
        if "CHIPS_ID" not in schema.names:
            print(f"  Skipping {fname} (no CHIPS_ID column)")
            continue
        table = pq.read_table(path, columns=["CHIPS_ID", "EMBEDDING"])
        chips_ids = table.column("CHIPS_ID").to_pylist()
        embeddings = table.column("EMBEDDING").to_pylist()
        for cid, e in zip(chips_ids, embeddings):
            if cid not in id_to_emb:
                id_to_emb[cid] = list(e) if not isinstance(e, list) else e
    print(f"Loaded {len(id_to_emb)} unique chips from parquets")
    return id_to_emb


def load_labels():
    """For each wanted category, collect positive chips. Returns {chips_id: category}."""
    # First pass: collect all positive labels per chip across wanted categories
    chip_cats = {}  # chips_id -> set of category names
    for cat in WANTED_CATEGORIES:
        labels_path = os.path.join(CONCEPT_LAB, cat, "labels.json")
        if not os.path.exists(labels_path):
            print(f"WARNING: no labels.json for {cat}")
            continue
        data = json.load(open(labels_path))
        labels = data.get("labels", {})
        for chips_id, label in labels.items():
            if label == "positive":
                chip_cats.setdefault(chips_id, set()).add(cat)

    # Drop multi-labelled chips
    result = {}
    dropped = 0
    for chips_id, cats in chip_cats.items():
        if len(cats) > 1:
            dropped += 1
            continue
        cat = next(iter(cats))
        result[chips_id] = DISPLAY_NAMES[cat]

    print(f"Labelled chips: {len(result)}, dropped {dropped} multi-labelled")
    for cat in sorted(WANTED_CATEGORIES):
        count = sum(1 for v in result.values() if v == DISPLAY_NAMES[cat])
        print(f"  {DISPLAY_NAMES[cat]}: {count}")
    return result


def load_concept_vectors():
    """Load all concept model weight vectors (not just wanted categories)."""
    vectors = {}
    for concept_dir in sorted(os.listdir(CONCEPT_LAB)):
        model_path = os.path.join(CONCEPT_LAB, concept_dir, "concept_model.json")
        if not os.path.exists(model_path):
            continue
        data = json.load(open(model_path))
        weights = data["model_params"]["weights"]
        assert len(weights) == 1024, f"{concept_dir} weights len={len(weights)}"
        # Use display name if it's a wanted category, otherwise clean up the folder name
        name = DISPLAY_NAMES.get(concept_dir, concept_dir.replace("_", " ").title())
        vectors[name] = [round(float(w), 6) for w in weights]
    print(f"Loaded {len(vectors)} concept vectors: {list(vectors.keys())}")
    return vectors


def main():
    sys.path.insert(0, "/Users/lgnd/Documents/repos/concept-lab/.venv/lib/python3.12/site-packages")

    id_to_emb = load_all_parquets(PARQUET_DIR)
    id_to_class = load_labels()
    concept_vectors = load_concept_vectors()

    # Build embeddings list (only chips we have both embedding and label for)
    embeddings = []
    missing = 0
    for chips_id, cls in sorted(id_to_class.items()):
        if chips_id not in id_to_emb:
            missing += 1
            continue
        emb = id_to_emb[chips_id]
        embeddings.append({
            "id": chips_id,
            "class": cls,
            "embedding": [round(float(x), 6) for x in emb],
        })

    if missing:
        print(f"WARNING: {missing} labelled chips not found in parquets")
    print(f"Final dataset: {len(embeddings)} points")

    os.makedirs(OUT_DIR, exist_ok=True)

    emb_path = os.path.join(OUT_DIR, "clay_embeddings.json")
    with open(emb_path, "w") as f:
        json.dump(embeddings, f)
    print(f"Wrote {emb_path} ({os.path.getsize(emb_path) / 1e6:.1f} MB)")

    ws_path = os.path.join(OUT_DIR, "clay_class_ws.json")
    with open(ws_path, "w") as f:
        json.dump(concept_vectors, f)
    print(f"Wrote {ws_path}")


if __name__ == "__main__":
    main()
