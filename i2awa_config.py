"""Paths and class lists for I2AwA (3D2 source, AwA2 target)."""
import os

DASA_ROOT = os.path.dirname(os.path.abspath(__file__))
I2AWA_ROOT = os.path.join(DASA_ROOT, "data", "I2AwA")

PATH_3D2 = os.path.join(I2AWA_ROOT, "I2AwA_source_domain", "3D2")
PATH_AWA2 = os.path.join(I2AWA_ROOT, "AwA2-data", "Animals_with_Attributes2", "JPEGImages")

FEATURE_DIR = os.path.join(I2AWA_ROOT, "features")
ATTR_DIR = os.path.join(I2AWA_ROOT, "attributes")
PSEUDO_DIR = os.path.join(I2AWA_ROOT, "pseudo")

SHR_NC = 40          # C_s: shared / known classes
UNK_NC = 10          # K:  number of UNKNOWN classes only (K-means on D_t^u)
TGT_NC = SHR_NC + UNK_NC  # C_t = C_s + K


def load_class_names():
    shr_path = os.path.join(I2AWA_ROOT, "shr_classes.txt")
    unk_path = os.path.join(I2AWA_ROOT, "unk_classes.txt")
    with open(shr_path, encoding="utf-8") as f:
        shared = [ln.strip() for ln in f if ln.strip()]
    with open(unk_path, encoding="utf-8") as f:
        unknown = [ln.strip() for ln in f if ln.strip()]
    return shared, unknown


def build_class_to_idx():
    shared, unknown = load_class_names()
    name_to_idx = {}
    for i, n in enumerate(shared):
        name_to_idx[n] = i
        name_to_idx[n.replace("+", " ")] = i
    for j, n in enumerate(unknown):
        name_to_idx[n] = SHR_NC + j
        name_to_idx[n.replace("+", " ")] = SHR_NC + j
    return name_to_idx, shared, unknown


def load_attribute_matrix():
    import numpy as np
    import pandas as pd

    path = os.path.join(I2AWA_ROOT, "att_bi.csv")
    df = pd.read_csv(path, index_col=0)
    shared, unknown = load_class_names()
    order = shared + unknown
    rows = []
    for name in order:
        key = name if name in df.index else name.replace("+", " ")
        if key not in df.index:
            for idx in df.index:
                if idx.replace("+", " ") == name.replace("+", " "):
                    key = idx
                    break
        rows.append(df.loc[key].values.astype(np.float32))
    return np.stack(rows, axis=0)


def feature_paths(domain, labeled_source=False):
    """domain: '3D2' or 'AwA2'"""
    os.makedirs(FEATURE_DIR, exist_ok=True)
    os.makedirs(ATTR_DIR, exist_ok=True)
    if domain == "3D2":
        prefix = "3D2"
    else:
        prefix = "AwA2"
    return {
        "feats": os.path.join(FEATURE_DIR, f"{prefix}_feats.csv"),
        "labels": os.path.join(FEATURE_DIR, f"{prefix}_labels.csv"),
        "att": os.path.join(ATTR_DIR, f"{prefix}_att_bi.csv"),
    }


def pseudo_path(src="3D2", tgt="AwA2"):
    os.makedirs(PSEUDO_DIR, exist_ok=True)
    return os.path.join(PSEUDO_DIR, f"{src}2{tgt}_sample_pseudo.csv")
