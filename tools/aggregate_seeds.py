#!/usr/bin/env python3
"""B2 helper: aggregate honest single-checkpoint test results across seeds.

Parses the `FINAL[seed=...] epoch=.. TEST k=v k=v ...` lines emitted by
train_msamba.py and prints mean +/- std for each metric.

Usage:
    python tools/aggregate_seeds.py runs/MSAmba_ALMT_mosi/seed_*.log
"""
import re
import sys
import glob
from collections import defaultdict

FINAL_RE = re.compile(r"FINAL\[seed=(?P<seed>[^\]]+)\]\s+epoch=(?P<epoch>\d+)\s+TEST\s+(?P<kv>.*)")


def parse_file(path):
    """Return the last FINAL record in a log file, or None."""
    last = None
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            m = FINAL_RE.search(line)
            if m:
                kv = {}
                for tok in m.group('kv').split():
                    if '=' in tok:
                        k, v = tok.split('=', 1)
                        try:
                            kv[k] = float(v)
                        except ValueError:
                            pass
                last = {'seed': m.group('seed'), 'epoch': int(m.group('epoch')), 'metrics': kv}
    return last


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    var = sum((x - mu) ** 2 for x in xs) / n if n > 1 else 0.0
    return mu, var ** 0.5


def main(argv):
    paths = []
    for a in argv:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        print("No log files matched.")
        return 1

    records = [r for r in (parse_file(p) for p in paths) if r]
    if not records:
        print("No FINAL[...] lines found in the given logs.")
        return 1

    print(f"Aggregating {len(records)} seed run(s):")
    for r in records:
        print(f"  seed={r['seed']:>6}  best_epoch={r['epoch']:>4}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in r['metrics'].items()))

    bucket = defaultdict(list)
    for r in records:
        for k, v in r['metrics'].items():
            bucket[k].append(v)

    print("\n==== mean +/- std over seeds (single-checkpoint, honest) ====")
    for k in sorted(bucket):
        mu, sd = mean_std(bucket[k])
        print(f"  {k:16s}: {mu:.4f} +/- {sd:.4f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
