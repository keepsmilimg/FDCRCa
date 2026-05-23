"""
I2AwA full pipeline: check data -> extract (if missing) -> K-means pseudo -> analysis -> optional Step3.

  python run_i2awa_pipeline.py --steps check,extract,pseudo,analysis
  python run_i2awa_pipeline.py --steps analysis,train --epochs 5
"""
import argparse
import os
import subprocess
import sys

from dataset_paths import feature_file, pseudo_file
from i2awa_config import I2AWA_ROOT, PATH_3D2, PATH_AWA2


def _exists(p):
    return os.path.isfile(p)


def step_check(args):
    print("=== Check paths ===")
    print("3D2 images:", PATH_3D2, os.path.isdir(PATH_3D2))
    print("AwA2 images:", PATH_AWA2, os.path.isdir(PATH_AWA2))
    for domain in ("src", "tgt"):
        for kind in ("feats", "labels", "att"):
            print(f"  {domain} {kind}:", _exists(feature_file(args, domain, kind)))
    print("  pseudo:", _exists(pseudo_file(args)))


def step_extract(args):
    py = sys.executable
    if not _exists(feature_file(args, "src", "feats")):
        subprocess.check_call([py, "extract_features_i2awa.py", "--domain", "3D2", "--batch_size", str(args.batch_size)])
    else:
        print("3D2 features exist, skip extract")
    if not _exists(feature_file(args, "tgt", "feats")):
        subprocess.check_call([py, "extract_features_i2awa.py", "--domain", "AwA2", "--batch_size", str(args.batch_size)])
    else:
        print("AwA2 features exist, skip extract")


def step_pseudo(args):
    if not _exists(pseudo_file(args)):
        subprocess.check_call([sys.executable, "init_pseudo_i2awa.py", "--k", str(args.tgt_nc)])
    else:
        print("Pseudo labels exist, skip init_pseudo")


def step_analysis(args):
    subprocess.check_call([
        sys.executable, "run_pseudo_label_analysis.py",
        "--data_path", I2AWA_ROOT,
        "--src", args.src, "--tgt", args.tgt,
        "--shr_nc", str(args.shr_nc), "--tgt_nc", str(args.tgt_nc),
    ])


def step_train(args):
    cmd = [sys.executable, "main.py", "--dataset", "I2AwA", "--step3", "train",
           "--epochs", str(args.epochs), "--batch_size", str(args.batch_size)]
    if args.pseudo_analysis:
        cmd.append("--pseudo_analysis")
    subprocess.check_call(cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=str, default="check,extract,pseudo,analysis",
                   help="comma-separated: check,extract,pseudo,analysis,train")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--pseudo_analysis", action="store_true")
    args_ns = p.parse_args()

    class A:
        dataset = "I2AwA"
        src, tgt = "3D2", "AwA2"
        data_path_source = data_path_target = I2AWA_ROOT
        shr_nc, tgt_nc, unk_nc = 40, 50, 10

    a = A()
    steps = [s.strip() for s in args_ns.steps.split(",")]
    a.epochs = args_ns.epochs
    a.batch_size = args_ns.batch_size
    a.pseudo_analysis = args_ns.pseudo_analysis

    for s in steps:
        if s == "check":
            step_check(a)
        elif s == "extract":
            step_extract(a)
        elif s == "pseudo":
            step_pseudo(a)
        elif s == "analysis":
            step_analysis(a)
        elif s == "train":
            step_train(a)
        else:
            print("Unknown step:", s)


if __name__ == "__main__":
    main()
