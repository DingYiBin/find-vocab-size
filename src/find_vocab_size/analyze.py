"""Analyze a dataset with the Qwen3-4B tokenizer to pick a suitable engram size.

Core idea: tokenize the whole dataset, build the frequency distribution of
n-grams (1/2/3 consecutive tokens), then report how many distinct n-grams
cover X% of all occurrences. Those "coverage vs. count" curves tell you
how large an engram table needs to be.

n-grams never cross document boundaries. To save memory, an n-gram
(id0, id1, ..., id_{n-1}) is packed into a single int:
    key = (...(id0 * V + id1) * V + ... ), V = tokenizer vocab size.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm import tqdm

DEFAULT_TOKENIZER = Path(__file__).resolve().parents[2] / "models" / "Qwen" / "Qwen3-4B" / "tokenizer.json"

# Coverage levels we report in the summary table.
COVERAGE_LEVELS = [0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999]

MAX_N = 4  # packing cost grows as V^n; keep <= 4 for a 151k vocab


@dataclass
class NGramStats:
    """Frequency stats for n-grams of one length n."""

    n: int
    total: int = 0  # total occurrences
    counter: Counter[int] = field(default_factory=Counter)  # packed key -> count

    @property
    def unique(self) -> int:
        return len(self.counter)


@dataclass
class Stats:
    n_docs: int
    tokenizer_vocab: int
    by_n: dict[int, NGramStats]  # n -> stats
    tok: object = field(repr=False, default=None)  # Tokenizer, kept for decoding top n-grams

    @property
    def total_tokens(self) -> int:
        return self.by_n[1].total

    @property
    def unique_tokens(self) -> int:
        return self.by_n[1].unique


def pack(ids: list[int], v: int) -> int:
    key = 0
    for i in ids:
        key = key * v + i
    return key


def unpack(key: int, n: int, v: int) -> list[int]:
    ids = [0] * n
    for i in range(n - 1, -1, -1):
        ids[i] = key % v
        key //= v
    return ids


def iter_objects(path: Path):
    """Yield raw documents (dict or str) from .jsonl / .jsonl.gz / .json / .txt files."""
    opener = gzip.open if path.suffix == ".gz" else open
    if path.suffix in {".jsonl", ".gz"}:
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:  # type: ignore[operator]
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    yield line  # fall back to raw text
                    continue
                yield obj
    elif path.suffix == ".json":
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:  # type: ignore[operator]
            data = json.load(f)
        if isinstance(data, dict):
            data = [data]
        for obj in data:
            yield obj
    elif path.suffix == ".txt":
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:  # type: ignore[operator]
            for line in f:
                yield line
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (use .jsonl/.jsonl.gz/.json/.txt)")


def _extract_text(obj, text_field: str | None) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if text_field:
            if text_field not in obj:
                return ""
            return str(obj[text_field])
        # No field specified: concatenate all string values (dict order).
        return "\n".join(str(v) for v in obj.values() if isinstance(v, str))
    return json.dumps(obj, ensure_ascii=False)


# sharegpt "from" -> chat "role"
SHAREGPT_ROLE = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant", "bing": "assistant"}
VALID_ROLES = {"user", "assistant", "system", "tool"}


def _to_messages(obj) -> list[dict] | None:
    """Convert a document to chat-template messages, or None if not a conversation.

    Supports:
      - {"conversations": [{"from": "human", "value": ...}]}   (sharegpt)
      - {"conversations": [{"role": "user", "content": ...}]}  (openai-style)
      - {"messages": [{"role": ..., "content": ...}]}           (openai-style)
    """
    if isinstance(obj, str):
        return [{"role": "user", "content": obj}]
    if not isinstance(obj, dict):
        return None
    raw = obj.get("conversations") if isinstance(obj.get("conversations"), list) else obj.get("messages")
    if not isinstance(raw, list):
        return None
    msgs = []
    for turn in raw:
        if not isinstance(turn, dict):
            continue
        if "content" in turn:  # openai-style turn
            role = str(turn.get("role", "user")).lower()
            role = SHAREGPT_ROLE.get(role, role)
            if role not in VALID_ROLES:
                role = "user"
            msgs.append({"role": role, "content": str(turn["content"])})
        elif "value" in turn:  # sharegpt-style turn
            role = SHAREGPT_ROLE.get(str(turn.get("from", "")).lower(), "user")
            msgs.append({"role": role, "content": str(turn["value"])})
    return msgs or None


def analyze(
    paths: Path | list[Path],
    tokenizer_path: Path,
    text_field: str | None,
    ngrams: list[int] = (1, 2, 3),
    batch_size: int = 1000,
    sample: float = 1.0,
    seed: int = 42,
    chat_template: bool = False,
    turns: str = "all",  # all | assistant | user
    apply_merge: Path | None = None,
) -> Stats:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    v = tok.get_vocab_size(with_added_tokens=True)
    stats = Stats(n_docs=0, tokenizer_vocab=v, by_n={n: NGramStats(n=n) for n in ngrams}, tok=tok)

    merge_lut: list[int] | None = None
    lut_np: np.ndarray | None = None
    if apply_merge is not None:
        data = json.loads(Path(apply_merge).read_text())
        # identity table, then overwrite merged ids -> canonical id;
        # list indexing beats dict.get by ~1.7x on large token volumes
        merge_lut = list(range(v))
        for k, x in data["map"].items():
            merge_lut[int(k)] = int(x)
        lut_np = np.asarray(merge_lut, dtype=np.int64)

    hf_tok = None
    if chat_template:
        from transformers import AutoTokenizer

        hf_tok = AutoTokenizer.from_pretrained(str(tokenizer_path.parent))

    def render_one(obj) -> list[str]:
        """Return the text(s) to tokenize for one document."""
        msgs = _to_messages(obj)
        if msgs is None:  # not a conversation -> treat whole doc text as one user turn
            msgs = [{"role": "user", "content": _extract_text(obj, text_field)}]
        if turns == "all":
            if hf_tok is None:
                return [_extract_text(obj, text_field)]
            return [hf_tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)]
        # Only one role's turns; each turn becomes its own document so
        # n-grams never span two turns.
        role = "assistant" if turns == "assistant" else "user"
        sel = [m["content"] for m in msgs if m["role"] == role and m.get("content")]
        if hf_tok is None:
            return sel
        return [
            hf_tok.apply_chat_template([{"role": role, "content": c}], tokenize=False, add_generation_prompt=False)
            for c in sel
        ]

    if isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    paths = [Path(p) for p in paths]

    def docs_iter():
        for p in paths:
            yield from iter_objects(p)

    rng = random.Random(seed)  # deterministic sampling
    n_skipped = 0
    pbar = tqdm(docs_iter(), desc="Tokenizing", unit="doc", file=sys.stderr)
    batch: list[str] = []
    for obj in pbar:
        texts = render_one(obj)
        if not texts:
            continue
        if sample < 1.0 and rng.random() > sample:
            n_skipped += 1
            continue
        batch.extend(texts)
        stats.n_docs += 1
        if len(batch) >= batch_size:
            _count_batch(tok, batch, stats, v, merge_lut, lut_np)
            batch = []
    if batch:
        _count_batch(tok, batch, stats, v, merge_lut, lut_np)
    if n_skipped:
        print(
            f"[sample] kept {stats.n_docs:,} docs, skipped {n_skipped:,} (rate={sample})",
            file=sys.stderr,
        )
    return stats


def _count_batch(tok, batch: list[str], stats: Stats, v: int, merge_lut, lut_np) -> None:
    """Batch-tokenize, vectorized-remap, vectorized-pack, then bulk-count.

    All encodings of the batch are concatenated into one int64 array with a
    -1 sentinel between documents (so n-grams never span documents), remapped
    through the LUT in one numpy op, packed into n-gram keys with vectorized
    multiply-adds, and counted with np.unique -- instead of a Python loop per
    n-gram position. n where v**n would overflow int64 falls back to the
    per-encoding Python path.
    """
    encs = tok.encode_batch(batch, add_special_tokens=False)
    if not encs:
        return

    total = sum(len(e.ids) for e in encs)
    if total == 0:
        return
    arr = np.empty(total + len(encs), dtype=np.int64)  # + 1 sentinel per doc
    pos = 0
    for e in encs:
        l = len(e.ids)
        arr[pos : pos + l] = e.ids
        pos += l
        arr[pos] = -1
        pos += 1

    if lut_np is not None:
        ok = arr >= 0
        arr[ok] = lut_np[arr[ok]]

    # sentinel count in window [w, w+n) == csum[w+n] - csum[w]
    sent = arr == -1
    csum = np.zeros(len(arr) + 1, dtype=np.int64)
    np.cumsum(sent, out=csum[1:])

    max_np_n = 1
    while v ** (max_np_n + 1) < 2**62:
        max_np_n += 1

    for n, ng in stats.by_n.items():
        m = len(arr) - n + 1
        if m <= 0:
            continue
        if n > max_np_n:  # int64 overflow -- slow path
            _count_n_python(encs, ng, v, merge_lut)
            continue
        keys = arr[:m].copy()
        for off in range(1, n):
            keys *= v
            keys += arr[off : off + m]
        valid = (csum[n : n + m] - csum[:m]) == 0  # window has no sentinel
        keys = keys[valid]
        if keys.size:
            uniq, cnt = np.unique(keys, return_counts=True)
            ng.counter.update(dict(zip(uniq.tolist(), cnt.tolist())))
            ng.total += int(keys.size)


def _count_n_python(encs, ng: NGramStats, v: int, lut: list[int] | None) -> None:
    """Fallback per-encoding counting for n too large for int64 packing."""
    remap = lut.__getitem__ if lut is not None else None
    for enc in encs:
        ids = list(map(remap, enc.ids)) if remap is not None else enc.ids
        if len(ids) < ng.n:
            continue
        ng.counter.update(pack(ids[i : i + ng.n], v) for i in range(len(ids) - ng.n + 1))
        ng.total += len(ids) - ng.n + 1


def coverage_table(ng: NGramStats) -> list[tuple[float, int]]:
    """Return [(coverage_ratio, n_grams_needed), ...] for each coverage level."""
    sorted_counts = sorted(ng.counter.values(), reverse=True)
    total = sum(sorted_counts)
    table: list[tuple[float, int]] = []
    cum = 0
    idx = 0
    for level in COVERAGE_LEVELS:
        target = total * level
        while idx < len(sorted_counts) and cum < target:
            cum += sorted_counts[idx]
            idx += 1
        table.append((level, idx))
    return table


def freq_buckets(ng: NGramStats) -> list[dict]:
    """Bucket distinct n-grams by occurrence-count ranges [2^k, 2^(k+1)).

    Complete coverage: buckets run from [1,2) up to the bucket containing
    the max count. For each bucket also return cumulative stats for
    "entries with count >= 2^k" (cum_ge / cum_occ), which is what
    min-count filtering decisions are based on.
    """
    distinct: Counter[int] = Counter()  # k -> #keys in bucket
    occ: Counter[int] = Counter()  # k -> sum of counts in bucket
    for c in ng.counter.values():
        k = max(c, 1).bit_length() - 1  # c in [2^k, 2^(k+1))
        distinct[k] += 1
        occ[k] += c
    if not distinct:
        return []
    kmax = max(distinct)
    cum_ge = {}  # k -> #entries with count >= 2^k
    cum_occ = {}  # k -> occurrences covered by those entries
    entries = 0
    total = 0
    for k in range(kmax, -1, -1):
        entries += distinct.get(k, 0)
        total += occ.get(k, 0)
        cum_ge[k] = entries
        cum_occ[k] = total
    return [
        {
            "lo": 1 << k,
            "hi": 1 << (k + 1),
            "distinct": distinct.get(k, 0),
            "occ": occ.get(k, 0),
            "cum_ge": cum_ge[k],
            "cum_occ": cum_occ[k],
        }
        for k in range(kmax + 1)
    ]


def report_freq_buckets(ng: NGramStats) -> None:
    print(f"\n  Frequency buckets: count in [2^k, 2^(k+1)), complete coverage")
    print(f"  {'bucket':>22} {'#distinct':>12} {'occurrences':>12} {'#count>=lo':>12} {'occ coverage':>13}")
    for r in freq_buckets(ng):
        cover = 100.0 * r["cum_occ"] / ng.total if ng.total else 0.0
        print(
            f"  [{r['lo']:>9,}, {r['hi']:>9,}) "
            f"{r['distinct']:>12,} {r['occ']:>12,} {r['cum_ge']:>12,} {cover:>12.2f}%"
        )


def report_ngram(ng: NGramStats, stats: Stats, top_k: int = 10) -> None:
    v = stats.tokenizer_vocab
    n = ng.n
    label = f"{n}-gram" + (" (token)" if n == 1 else " (consecutive tokens)" if n == 2 else "s")
    print(f"\n=== {label} ===")
    print(f"total occurrences: {ng.total:,}")
    print(f"unique {n}-grams:  {ng.unique:,}")
    hapax = sum(1 for c in ng.counter.values() if c == 1)
    print(f"appear only once: {hapax:,} ({100.0 * hapax / max(ng.unique, 1):.1f}% of unique)")

    print(f"\n  Coverage: #distinct {n}-grams needed to cover X% of occurrences")
    print(f"  {'coverage':>10} {'#ngrams':>14}")
    for level, cnt in coverage_table(ng):
        print(f"  {level * 100:>9.1f}% {cnt:>14,}")

    print(f"\n  Top {top_k} most frequent {n}-grams:")
    for key, cnt in ng.counter.most_common(top_k):
        ids = unpack(key, n, v)
        text = stats.tok.decode(ids) if stats.tok is not None else str(ids)
        freq = cnt / ng.total if ng.total else 0.0
        print(f"    {cnt:>12,}  ({freq * 100:.2f}%)  {text!r}")

    report_freq_buckets(ng)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="find-vocab-size",
        description="Analyze token/n-gram coverage of a dataset to choose an engram size (Qwen3-4B tokenizer).",
    )
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="+",
        help="One or more dataset files: .jsonl / .jsonl.gz / .json / .txt (stats are combined)",
    )
    parser.add_argument("--text-field", default=None, help="JSON field holding the text (default: concat all string fields)")
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER, help="Path to tokenizer.json")
    parser.add_argument(
        "--ngrams",
        default="1,2,3",
        help="Comma-separated n-gram lengths to analyze (default: 1,2,3; max 4)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of documents to keep (0<p<=1, default: 1.0 = all). For large datasets, memory scales with sample size",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for document sampling (default: 42)")
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help="Render conversations with the Qwen3 chat template before tokenizing "
        "(supports sharegpt 'conversations' and openai 'messages' formats)",
    )
    parser.add_argument(
        "--turns",
        choices=["all", "assistant", "user"],
        default="all",
        help="Which turns to analyze: all turns, or only assistant (response) / user turns. "
        "With a turn filter each turn is a separate document, so n-grams never span turns "
        "(default: all)",
    )
    parser.add_argument(
        "--apply-merge",
        type=Path,
        default=None,
        help="Path to vocab_merge_map.json (from find-vocab-merge): remap token ids to "
        "canonical ids BEFORE counting, so all stats reflect the compressed key space",
    )
    args = parser.parse_args(argv)

    missing = [p for p in args.dataset if not p.exists()]
    if missing:
        parser.error(f"Dataset file(s) not found: {', '.join(str(p) for p in missing)}")
    if not args.tokenizer.exists():
        parser.error(f"Tokenizer not found: {args.tokenizer}")
    if not 0 < args.sample <= 1:
        parser.error("--sample must be in (0, 1]")
    if args.apply_merge is not None and not args.apply_merge.exists():
        parser.error(f"--apply-merge map not found: {args.apply_merge}")

    try:
        ngrams = sorted({int(x) for x in args.ngrams.split(",")})
        bad = [n for n in ngrams if not 1 <= n <= MAX_N]
        if bad:
            raise ValueError(str(bad))
    except ValueError:
        parser.error(f"--ngrams must be comma-separated ints in [1, {MAX_N}]")

    stats = analyze(
        args.dataset,
        args.tokenizer,
        args.text_field,
        ngrams,
        sample=args.sample,
        seed=args.seed,
        chat_template=args.chat_template,
        turns=args.turns,
        apply_merge=args.apply_merge,
    )

    if len(args.dataset) == 1:
        print(f"\n=== Dataset: {args.dataset[0].name} ===")
    else:
        names = ", ".join(p.name for p in args.dataset)
        if len(names) > 100:
            names = names[:97] + "..."
        print(f"\n=== Datasets ({len(args.dataset)} files, combined): {names} ===")
    if args.chat_template:
        print("chat template:  APPLIED (Qwen3 template, add_generation_prompt=False)")
    if args.apply_merge is not None:
        n_merged = len(json.loads(args.apply_merge.read_text())["map"])
        print(
            f"vocab merge:    APPLIED ({n_merged:,} ids remapped -> "
            f"effective vocab {stats.tokenizer_vocab - n_merged:,})"
        )
    if args.turns != "all":
        print(f"turns filter:  {args.turns} only (each turn = one document)")
    if args.sample < 1.0:
        print(f"SAMPLED:       {args.sample:.1%} of documents (seed={args.seed}) -- counts below are from the sample only")
    print(f"documents:      {stats.n_docs:,}")
    print(f"total tokens:   {stats.total_tokens:,}")
    print(f"tokenizer vocab: {stats.tokenizer_vocab:,}")
    print(f"unique tokens:  {stats.unique_tokens:,}")

    for n in ngrams:
        report_ngram(stats.by_n[n], stats)

    print("\n=== Engram size suggestions (rounded up to power of 2) ===")
    for n in ngrams:
        ng = stats.by_n[n]
        for level, cnt in coverage_table(ng):
            if level in (0.9, 0.95, 0.99):
                pow2 = 1 << (cnt - 1).bit_length() if cnt > 0 else 0
                print(f"  {n}-gram cover {level * 100:.0f}%: {cnt:,} entries -> {pow2:,}")


if __name__ == "__main__":
    main()