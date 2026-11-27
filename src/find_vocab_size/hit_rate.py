"""Hit-rate analysis: does an n-gram inventory built from a reference corpus
cover the n-grams of model-generated responses?

Two inputs:
  1. Reference corpus -- one or more dataset files, counted with the SAME
     pipeline/options as find-vocab-size (chat template, turns filter,
     vocab-merge LUT). This is "what the engram contains".
  2. Eval file -- a jsonl where each line's `content` field (configurable)
     is a model response. Responses are tokenized DIRECTLY (no chat template,
     no conversation parsing), each line as one document.

For every 2-gram/3-gram occurrence in the responses we look up its count in
the reference inventory and report hit rates at several count thresholds
(>=1/2/4/8/16), a per-response hit-rate distribution, and the most frequent
missed n-grams.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from find_vocab_size.analyze import Stats, analyze

DEFAULT_TOKENIZER = Path(__file__).resolve().parents[2] / "models" / "Qwen" / "Qwen3-4B" / "tokenizer.json"
HIT_THRESHOLDS = [1, 2, 4, 8, 16]


def packed_keys(ids: np.ndarray, n: int, v: int) -> np.ndarray:
    m = len(ids) - n + 1
    if m <= 0:
        return np.empty(0, dtype=np.int64)
    keys = ids[:m].copy()
    for off in range(1, n):
        keys = keys * v + ids[off : off + m]
    return keys


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}%" if x is not None else "-"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="find-vocab-hit",
        description="Check how well n-grams from a reference corpus cover model responses "
        "(built on the find-vocab-size pipeline).",
    )
    parser.add_argument("refs", type=Path, nargs="*", help="Reference dataset file(s) to build the n-gram inventory from (or use --load)")
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        metavar="INVENTORY.npz",
        help="Load a pre-built inventory (from find-vocab-size --dump) instead of rebuilding from refs",
    )
    parser.add_argument("--eval", type=Path, required=True, help="Eval jsonl whose FIELD (default: content) holds model responses")
    parser.add_argument("--field", default="content", help="JSON field holding the response text (default: content)")
    parser.add_argument("--id-field", default="unique_id", help="Field identifying each response in reports (default: unique_id)")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER, help="Path to tokenizer.json")
    parser.add_argument("--chat-template", action="store_true", help="Build the REFERENCE with the Qwen3 chat template (same as find-vocab-size)")
    parser.add_argument("--turns", choices=["all", "assistant", "user"], default="assistant", help="Which turns the REFERENCE counts (default: assistant)")
    parser.add_argument("--apply-merge", type=Path, default=None, help="vocab_merge_map.json applied to BOTH reference and responses")
    parser.add_argument("--ngrams", default="2,3", help="Comma-separated n-gram lengths (default: 2,3)")
    parser.add_argument("--worst", type=int, default=5, help="How many worst responses to list (default: 5)")
    parser.add_argument("--miss-top", type=int, default=15, help="How many missed n-grams to list per level (default: 15)")
    args = parser.parse_args(argv)

    for p in args.refs:
        if not p.exists():
            parser.error(f"Reference file not found: {p}")
    if not args.eval.exists():
        parser.error(f"Eval file not found: {args.eval}")
    if not args.tokenizer.exists():
        parser.error(f"Tokenizer not found: {args.tokenizer}")
    if args.load is not None and args.refs:
        parser.error("Provide either reference file(s) or --load INVENTORY.npz, not both")
    if args.load is None and not args.refs:
        parser.error("Provide reference file(s) or --load INVENTORY.npz")

    try:
        ngrams = sorted({int(x) for x in args.ngrams.split(",")})
        if not ngrams or any(n < 2 or n > 3 for n in ngrams):
            raise ValueError
    except ValueError:
        parser.error("--ngrams must be comma-separated ints in [2, 3]")

    # ---- 1. reference inventory: build from refs, or load a saved one ----
    if args.load is not None:
        from find_vocab_size.analyze import load_stats

        ref_stats, ref_meta = load_stats(args.load)
        # consistency checks against the loaded build options
        inv_merge = ref_meta.get("merge")
        want_merge = str(args.apply_merge) if args.apply_merge else None
        if inv_merge != want_merge:
            parser.error(
                f"--apply-merge mismatch with inventory: built with merge={inv_merge}, "
                f"you passed {want_merge}. Keys would not be comparable; align the flags."
            )
        for n in ngrams:
            if n not in ref_stats.by_n:
                parser.error(f"Inventory lacks {n}-gram stats (built with ngrams={ref_meta.get('ngrams')})")
    else:
        # include n=1 so Stats.total_tokens works; only requested n are reported
        ref_stats: Stats = analyze(
            args.refs,
            args.tokenizer,
            text_field=None,
            ngrams=sorted({1, *ngrams}),
            chat_template=args.chat_template,
            turns=args.turns,
            apply_merge=args.apply_merge,
        )
        ref_meta = None

    # ---- 2. tokenize eval responses directly ----
    tok = Tokenizer.from_file(str(args.tokenizer))
    v = tok.get_vocab_size(with_added_tokens=True)
    lut_np = None
    if args.apply_merge is not None:
        data = json.loads(Path(args.apply_merge).read_text())
        lut = list(range(v))
        for k, x in data["map"].items():
            lut[int(k)] = int(x)
        lut_np = np.asarray(lut, dtype=np.int64)

    eval_counter: dict[int, Counter[int]] = {n: Counter() for n in ngrams}
    # per-n occurrences and hits at each threshold
    occ = {n: 0 for n in ngrams}
    hits = {n: {t: 0 for t in HIT_THRESHOLDS} for n in ngrams}
    # distribution of hits across REFERENCE frequency buckets [2^k, 2^(k+1))
    bucket_hits: dict[int, Counter[int]] = {n: Counter() for n in ngrams}  # k -> eval occurrences
    miss_occ = {n: 0 for n in ngrams}
    per_resp: dict[int, list[tuple[str, int, int, int]]] = {n: [] for n in ngrams}  # (uid, occ, hit>=1, hit>=2)
    n_resp = 0
    n_resp_tokens = 0
    n_bad = 0

    batch: list[str] = []
    uids: list[str] = []

    def flush() -> None:
        nonlocal n_resp, n_resp_tokens
        if not batch:
            return
        encs = tok.encode_batch(batch, add_special_tokens=False)
        for uid, enc in zip(uids, encs):
            ids = np.asarray(enc.ids, dtype=np.int64)
            if lut_np is not None:
                ids = lut_np[ids]
            n_resp_tokens += len(ids)
            for n in ngrams:
                keys = packed_keys(ids, n, v)
                ref_get = ref_stats.by_n[n].counter.get
                klist = keys.tolist()
                counts = [ref_get(k, 0) for k in klist]
                occ[n] += len(klist)
                h1 = sum(1 for c in counts if c >= 1)
                h2 = sum(1 for c in counts if c >= 2)
                for t in HIT_THRESHOLDS:
                    hits[n][t] += sum(1 for c in counts if c >= t)
                bucket_hits[n].update(c.bit_length() - 1 for c in counts if c > 0)
                miss_occ[n] += len(klist) - h1
                eval_counter[n].update(klist)
                if klist:
                    per_resp[n].append((uid, len(klist), h1, h2))
        batch.clear()
        uids.clear()

    with open(args.eval, encoding="utf-8", errors="replace") as f:
        for line in tqdm(f, desc="Eval responses", unit="resp", file=sys.stderr):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            text = obj.get(args.field) if isinstance(obj, dict) else None
            if not text:
                n_bad += 1
                continue
            uid = str(obj.get(args.id_field, f"line{n_resp}"))
            batch.append(str(text))
            uids.append(uid)
            n_resp += 1
            if len(batch) >= 1000:
                flush()
    flush()

    # ---- 3. report ----
    print(f"\n=== Reference inventory (engram source) ===")
    if args.load is not None:
        print(f"inventory:       {args.load.name} (pre-built)")
        print(f"built from:      {', '.join(Path(f).name for f in ref_meta.get('files', [])) if ref_meta.get('files') else 'n/a'}")
        print(f"options:         chat_template={ref_meta.get('chat_template')}, turns={ref_meta.get('turns')}, merge={ref_meta.get('merge')}")
        print(f"documents:       {ref_meta.get('n_docs', 0):,}")
        print(f"total tokens:    {ref_meta.get('total_tokens', 0):,}")
    else:
        print(f"files:           {len(args.refs)} ({', '.join(p.name for p in args.refs) if len(args.refs) <= 5 else '...'})")
        print(f"options:         chat_template={args.chat_template}, turns={args.turns}, merge={bool(args.apply_merge)}")
        print(f"documents:       {ref_stats.n_docs:,}")
        print(f"total tokens:    {ref_stats.total_tokens:,}")
    for n in ngrams:
        print(f"unique {n}-grams:  {ref_stats.by_n[n].unique:,}")

    print(f"\n=== Eval responses ===")
    print(f"file:            {args.eval.name} (field: {args.field})")
    print(f"responses:       {n_resp:,}  (skipped {n_bad:,} empty/invalid lines)")
    print(f"total tokens:    {n_resp_tokens:,}")

    print(f"\n=== Hit rates: share of response n-gram occurrences present in reference ===")
    header = "".join(f"  hit@>={t:<3}" for t in HIT_THRESHOLDS)
    print(f"{'n':>3} {'occurrences':>14}{header}")
    for n in ngrams:
        row = f"{n:>3} {occ[n]:>14,}"
        for t in HIT_THRESHOLDS:
            rate = hits[n][t] / occ[n] if occ[n] else None
            row += f"  {fmt_pct(rate):>8}"
        print(row)

    print(f"\n=== Hit distribution by REFERENCE frequency bucket (count in [2^k, 2^(k+1))) ===")
    for n in ngrams:
        if occ[n] == 0:
            continue
        bc = bucket_hits[n]
        kmax = max(bc) if bc else -1
        print(f"  n={n}  ({occ[n]:,} eval occurrences):")
        print(f"  {'ref bucket':>26} {'eval occ':>10} {'% of eval':>10} {'cum hit@>=lo':>13}")
        print(f"  {'missed (ref count = 0)':>26} {miss_occ[n]:>10,} {fmt_pct(miss_occ[n] / occ[n]):>10}")
        cum = occ[n] - miss_occ[n]  # occurrences with ref count >= 1
        for k in range(kmax + 1):
            c_k = bc.get(k, 0)
            lo = 1 << k
            print(
                f"  [{lo:>9,}, {lo << 1:>9,}) {c_k:>10,} {fmt_pct(c_k / occ[n]):>10} {fmt_pct(cum / occ[n]):>13}"
            )
            cum -= c_k

    for n in ngrams:
        if not per_resp[n]:
            continue
        rates1 = [h1 / o for _, o, h1, _ in per_resp[n]]
        print(f"\n=== Per-response hit rate, {n}-gram (hit@>=1) ===")
        print(f"mean {fmt_pct(statistics.mean(rates1))}  p50 {fmt_pct(statistics.median(rates1))}  "
              f"min {fmt_pct(min(rates1))}  max {fmt_pct(max(rates1))}")
        worst = sorted(per_resp[n], key=lambda r: r[2] / max(r[1], 1))[: args.worst]
        for uid, o, h1, h2 in worst:
            print(f"  worst: {uid}  hit@>=1 {fmt_pct(h1 / o if o else None)}  hit@>=2 {fmt_pct(h2 / o if o else None)}  ({h1:,}/{o:,})")

    print(f"\n=== Most frequent MISSED n-grams (in responses, absent from reference) ===")
    for n in ngrams:
        ref_counter = ref_stats.by_n[n].counter
        missed = [(k, c) for k, c in eval_counter[n].most_common() if ref_counter.get(k, 0) == 0][: args.miss_top]
        if not missed:
            print(f"  {n}-gram: none")
            continue
        print(f"  {n}-gram ({len(missed)} shown):")
        for k, c in missed:
            ids = []
            key = k
            for _ in range(n):
                ids.append(key % v)
                key //= v
            text = tok.decode(list(reversed(ids)))
            print(f"    {c:>8,}x  {text!r}")


if __name__ == "__main__":
    main()