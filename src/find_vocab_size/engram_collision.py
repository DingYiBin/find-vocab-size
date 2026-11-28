"""Collision analysis for the DSpark Engram memory (hash_version xor_v1).

Simulates the Engram hash on every distinct n-gram of a find-vocab-size
inventory (.npz) and measures how many n-grams collide into the same
bucket of each head's embedding table, weighted by occurrence counts.

Hash semantics (must mirror specforge EngramMemory.bucket_ids):
  bigram  mixed = (cur * m0) ^ (prev * m1)          -> heads 0..H/2-1
  trigram mixed = bigram ^ (older * m2)             -> heads H/2..H-1
  bucket = mixed % C   (mixed is non-negative int64: products < 1.3e13)

Inventory packing (find_vocab_size.analyze.pack) is (older, prev, cur) in
reading order, so unpacking gives exactly the hash inputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_TOKENIZER = Path(__file__).resolve().parents[2] / "models" / "Qwen" / "Qwen3-4B" / "tokenizer.json"


def find_engram_config(node):
    """Locate the engram options dict inside a SpecForge-style config JSON."""
    if isinstance(node, dict):
        if node.get("hash_version") == "xor_v1" and "bucket_counts" in node:
            return node
        for key in ("engram", "dflash_config"):
            if key in node:
                found = find_engram_config(node[key])
                if found is not None:
                    return found
    return None


def unpack_ids(keys: np.ndarray, n: int, v: int) -> list[np.ndarray]:
    """packed keys -> [older, prev, cur] reading-order id arrays."""
    cols = []
    rem = keys
    for _ in range(n):
        rem, col = np.divmod(rem, v)
        cols.append(col)
    cols.reverse()  # now [t0(oldest), ..., t_{n-1}(cur)]
    return cols


def mixed_values(cols: list[np.ndarray], multipliers: list[int]) -> np.ndarray:
    m0, m1, m2 = multipliers
    cur, prev = cols[-1], cols[-2]
    mixed = (cur * m0) ^ (prev * m1)
    if len(cols) >= 3:
        mixed = mixed ^ (cols[-3] * m2)
    return mixed


def head_metrics(mixed: np.ndarray, counts: np.ndarray, C: int) -> dict:
    bucket = mixed % C
    n = int(len(mixed))
    distinct_per = np.bincount(bucket, minlength=C)
    mass_per = np.bincount(bucket, weights=counts.astype(np.float64), minlength=C)
    total_mass = float(counts.sum())
    shared = distinct_per > 1
    p = mass_per / total_mass if total_mass else mass_per
    q = counts.astype(np.float64) / total_mass if total_mass else counts.astype(np.float64)
    return {
        "C": C,
        "n": n,
        "load": n / C,
        "occupied": int((distinct_per > 0).sum()),
        "distinct_at_risk": int(distinct_per[shared].sum()),
        "mass_at_risk": float(mass_per[shared].sum()),
        "p_same_bucket": float((p * p).sum()),
        "p_distinct_pair": float((p * p).sum() - (q * q).sum()),
        "max_bucket": int(distinct_per.max()),
        "bucket": bucket,
        "distinct_per": distinct_per,
        "mass_per": mass_per,
    }


def collision_freq_stats(mixed: np.ndarray, counts: np.ndarray, C: int) -> dict:
    """Frequency-weighted collision damage stats for one head.

    contamination(k) = 1 - count_k / mass(bucket_k): share of an n-gram's
    bucket mass that belongs to OTHER n-grams -- occurrence-weighted this is
    the expected dilution a random hit suffers.
    """
    bucket = mixed % C
    own = counts.astype(np.float64)
    mass_per = np.bincount(bucket, weights=own, minlength=C)
    bmass = mass_per[bucket]
    contam = 1.0 - own / bmass
    weighted_contam = float((contam * own).sum() / own.sum())

    # top-1 dominance: share of bucket mass held by its largest n-gram
    order = np.lexsort((-own, bucket))
    bs = bucket[order]
    starts = np.nonzero(np.r_[True, bs[1:] != bs[:-1]])[0]
    top1 = np.zeros(C)
    top1[bs[starts]] = own[order][starts]
    top1_share = top1[bucket] / bmass
    # weighted p50 of top1_share over occurrences
    w = own / own.sum()
    srt = np.argsort(top1_share)
    cw = np.cumsum(w[srt])
    p50 = float(top1_share[srt][np.searchsorted(cw, 0.5)])
    mean_top1 = float((top1_share * own).sum() / own.sum())

    # neighbor mass distribution: other-mass = bmass - own, occurrence-weighted
    other = bmass - own
    kmax = int(int(other.max())).bit_length() if other.size else 0
    hist = []
    for k in range(kmax + 1):
        lo = (1 << k) - 1 if k else 0  # bucket [0,1) = no neighbor
        hi = 1 << k if k else 1
        m = (other >= lo) & (other < hi) if k else (other == 0)
        hist.append((k, float(own[m].sum())))

    # worst buckets by SECOND-largest member count (high-freq x high-freq)
    second = np.zeros(C)
    has_second = np.diff(starts, append=len(bs)) > 1
    second[bs[starts][has_second]] = own[order][starts[has_second] + 1]
    worst_second = np.argsort(-second)[:10]
    return {
        "bucket": bucket,
        "weighted_contam": weighted_contam,
        "mean_top1": mean_top1,
        "p50_top1": p50,
        "hist": hist,
        "second": second,
        "worst_second": [int(b) for b in worst_second if second[b] > 0],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="find-engram-collision",
        description="Check Engram (xor_v1) bucket collisions using a find-vocab-size inventory.",
    )
    parser.add_argument("inventory", type=Path, help=".npz from find-vocab-size --dump")
    parser.add_argument("--config", type=Path, required=True, help="Model config JSON containing the engram options (or a bare engram JSON)")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER, help="Path to tokenizer.json (for decoding examples)")
    parser.add_argument("--top-buckets", type=int, default=5, help="How many worst buckets to detail per order (default: 5)")
    parser.add_argument("--examples", type=int, default=5, help="Example n-grams to show per worst bucket (default: 5)")
    args = parser.parse_args(argv)

    if not args.inventory.exists():
        parser.error(f"Inventory not found: {args.inventory}")
    if not args.config.exists():
        parser.error(f"Config not found: {args.config}")

    cfg_raw = json.loads(args.config.read_text())
    cfg = find_engram_config(cfg_raw)
    if cfg is None:
        parser.error("No engram section (hash_version=xor_v1 + bucket_counts) found in --config")
    heads_per_order = cfg.get("heads_per_order", 2)
    bucket_counts = list(cfg["bucket_counts"])
    multipliers = list(cfg["hash_multipliers"])
    if len(bucket_counts) != 2 * heads_per_order:
        parser.error(f"bucket_counts must have 2*heads_per_order entries (got {len(bucket_counts)})")

    data = np.load(args.inventory, allow_pickle=False)
    meta = json.loads(data["meta"].tobytes().decode("utf-8"))
    v = meta["vocab_size"]
    if "keys_2" not in data.files:
        parser.error("Inventory lacks 2-gram stats; rebuild with --ngrams 2,3")
    if "keys_3" not in data.files:
        parser.error("Inventory lacks 3-gram stats; rebuild with --ngrams 2,3")

    tok = None
    if args.tokenizer.exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(args.tokenizer))

    print("=== Engram config ===")
    print(f"heads_per_order: {heads_per_order}  head_dim: {cfg.get('head_dim')}")
    print(f"multipliers:     {multipliers}")
    print(f"bucket_counts:  bigram {bucket_counts[:heads_per_order]}")
    print(f"                 trigram {bucket_counts[heads_per_order:]}")
    print(f"\n=== Inventory ===")
    print(f"files:          {', '.join(Path(f).name for f in meta.get('files', [])) or 'n/a'}")
    print(f"options:        chat_template={meta.get('chat_template')}, turns={meta.get('turns')}, merge={meta.get('merge')}")
    print(f"total tokens:   {meta.get('total_tokens', 0):,}")

    orders = [
        ("bigram (2-gram)", 2, bucket_counts[:heads_per_order]),
        ("trigram (3-gram)", 3, bucket_counts[heads_per_order:]),
    ]
    for label, n, counts_c in orders:
        keys = data[f"keys_{n}"]
        counts = data[f"counts_{n}"].astype(np.int64)
        cols = unpack_ids(keys, n, v)
        mixed = mixed_values(cols, multipliers)

        pre_distinct = len(np.unique(mixed))
        print(f"\n=== {label} inventory: {len(keys):,} distinct, {int(counts.sum()):,} occurrences ===")
        print(
            f"pre-hash collisions (identical mixed value -> collide in EVERY head): "
            f"{len(keys) - pre_distinct:,} ({100.0 * (len(keys) - pre_distinct) / max(len(keys), 1):.3f}%)"
        )
        print(
            f"{'head':>4} {'C':>10} {'load':>7} {'occupied':>10} "
            f"{'distinct@risk':>14} {'occ@risk':>10} {'P(pair)':>9} {'max/bkt':>8}"
        )
        per_head = []
        for h, C in enumerate(counts_c):
            m = head_metrics(mixed, counts, C)
            per_head.append(m)
            print(
                f"{h:>4} {C:>10,} {m['load']:>7.3f} {m['occupied']:>10,} "
                f"{100.0 * m['distinct_at_risk'] / max(m['n'], 1):>13.2f}% "
                f"{100.0 * m['mass_at_risk'] / max(float(counts.sum()), 1):>9.2f}% "
                f"{m['p_same_bucket']:>9.4f} {m['max_bucket']:>8,}"
            )
        print("  distinct@risk: % of distinct n-grams sharing a bucket with another n-gram")
        print("  occ@risk:      % of occurrences landing in a bucket shared by >1 n-gram")
        print("  P(pair):       probability two occurrence-weighted draws hit the same bucket")

        # frequency-weighted collision damage (head 0; heads differ only by C)
        fs = collision_freq_stats(mixed, counts, counts_c[0])
        print(f"\n  === Collision frequency stats ({label.split()[0]}, head 0) ===")
        print(f"  weighted contamination:  {fs['weighted_contam'] * 100:.2f}%   "
              f"(expected share of bucket mass belonging to OTHER n-grams, per occurrence)")
        print(f"  top-1 dominance:         mean {fs['mean_top1'] * 100:.2f}%  "
              f"p50 {fs['p50_top1'] * 100:.2f}%   (share of bucket mass held by its largest n-gram)")
        print(f"\n  Neighbor-mass distribution (other n-grams' mass in the hit bucket, occurrence-weighted):")
        total_mass = float(counts.sum())
        nonzero_k = [k for k, _ in fs["hist"] if k > 0]
        for k, m_k in fs["hist"]:
            if m_k <= 0 and (k > 0 or not nonzero_k):
                continue
            tag = "no collision (own bucket)" if k == 0 else f"other mass [{1 << (k-1):,}, {1 << k:,})"
            print(f"    {tag:>34} {m_k:>16,.0f}  {100.0 * m_k / total_mass:>7.2f}% of occurrences")

        if tok is not None and fs["worst_second"]:
            print(f"\n  Worst high-freq x high-freq buckets (ranked by 2nd-largest member count):")
            for b in fs["worst_second"][: args.top_buckets]:
                members = np.where(fs["bucket"] == b)[0]
                members = members[np.argsort(-counts[members])]
                print(f"    bucket {b:,} (2nd member: {int(fs['second'][b]):,}x):")
                for idx in members[: args.examples]:
                    ids = [int(c[idx]) for c in cols]
                    print(f"      {int(counts[idx]):>11,}x  {tok.decode(ids)!r}")


        # worst buckets for the FIRST head of this order (heads differ only by C)
        if tok is not None and args.top_buckets > 0:
            m = per_head[0]
            order_by_distinct = np.argsort(-m["distinct_per"], kind="stable")[: args.top_buckets]
            print(f"\n  Worst buckets of {label.split()[0]} head 0 (by #distinct n-grams):")
            for b in order_by_distinct:
                members = np.where(m["bucket"] == b)[0]
                print(f"    bucket {int(b):,}: {m['distinct_per'][b]:,} distinct, mass {int(m['mass_per'][b]):,}")
                shown = 0
                for idx in members:
                    if shown >= args.examples:
                        break
                    ids = [int(c[idx]) for c in cols]
                    text = tok.decode(ids)
                    print(f"      {int(counts[idx]):>9,}x  {text!r}")
                    shown += 1
    if tok is None:
        print("\n(tokenizer not found; bucket examples skipped)")


if __name__ == "__main__":
    main()