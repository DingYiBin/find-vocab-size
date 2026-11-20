"""Inspect the Qwen3-4B vocab for mergeable token ids.

Motivation (Engram-style vocab compression): before building an engram,
map semantically-equivalent tokens to one canonical id so the key space
shrinks. This script finds merge candidates and can emit the id->id map.

Merge tiers (increasingly aggressive, applied via union-find):
  T1  case-only:        "The" == "the" == "THE"
  T2  whitespace kind:  tab == space, CR == LF   (within each ws class, run lengths kept)
  T3  whitespace run:   "  the" == " the", "\n\n" == "\n"  (runs collapsed per class)

Whitespace classes (kept distinct from each other):
  space family:   0x20 space, 0x09 tab
  newline family: 0x0a LF, 0x0d CR, 0x0b VT, 0x0c FF

Tokens are byte-level BPE strings; we decode each vocab entry to its raw
bytes with the standard GPT-2 byte->unicode table (Qwen uses the same)
and compare at the byte level, so partial UTF-8 fragments are handled.
Added/special tokens (e.g. <|im_start|>) are never merged.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

DEFAULT_TOKENIZER = Path(__file__).resolve().parents[2] / "models" / "Qwen" / "Qwen3-4B" / "tokenizer.json"

# whitespace is split into two semantic classes; families never merge with each other
WS_SPACE = b" \t"  # 0x20 space, 0x09 tab -> canonical 0x20
WS_NEWLINE = b"\n\r\x0b\x0c"  # LF, CR, VT, FF -> canonical 0x0a

_WS_TABLE = bytes(bytearray(range(256)))
for _b in WS_SPACE:
    _WS_TABLE = _WS_TABLE[:_b] + bytes([0x20]) + _WS_TABLE[_b + 1 :]
for _b in WS_NEWLINE:
    _WS_TABLE = _WS_TABLE[:_b] + bytes([0x0A]) + _WS_TABLE[_b + 1 :]


def bytes_to_unicode() -> dict[int, str]:
    """Standard GPT-2 byte->unicode table (used by Qwen byte-level BPE)."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


BYTE_DECODER = {v: bytes([k]) for k, v in bytes_to_unicode().items()}


def token_to_bytes(token_str: str) -> bytes:
    return b"".join(BYTE_DECODER[ch] for ch in token_str)


def norm_case(bs: bytes) -> bytes:
    return bs.lower()


def norm_ws_kind(bs: bytes) -> bytes:
    """T2: fold ws kinds within each class (tab->space, CR->LF), keep runs; then lowercase."""
    return bs.translate(_WS_TABLE).lower()


def norm_ws_run(bs: bytes) -> bytes:
    """T3: additionally collapse consecutive runs of the same ws char to one."""
    folded = norm_ws_kind(bs)
    out = bytearray()
    prev_ws = -1
    for byte in folded:
        if byte in (0x20, 0x0A):
            if byte != prev_ws:
                out.append(byte)
            prev_ws = byte
        else:
            out.append(byte)
            prev_ws = -1
    return bytes(out)


def show(bs: bytes) -> str:
    return repr(bs.decode("utf-8", errors="replace"))


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def main(argv: list[str] | None = None) -> None:
    from tokenizers import Tokenizer

    parser = argparse.ArgumentParser(
        prog="find-vocab-merge",
        description="Find mergeable (semantically close) token ids in the Qwen3-4B vocab and emit an id->id mapping.",
    )
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER, help="Path to tokenizer.json")
    parser.add_argument("--min-group", type=int, default=2, help="Only report groups with >= this many members")
    parser.add_argument("--show-groups", type=int, default=20, help="How many example groups to print")
    parser.add_argument("--out", type=Path, default=None, help="Write id->id mapping JSON to this path")
    args = parser.parse_args(argv)

    tok = Tokenizer.from_file(str(args.tokenizer))
    base_vocab: dict[str, int] = tok.get_vocab(with_added_tokens=False)
    full_vocab: dict[str, int] = tok.get_vocab(with_added_tokens=True)
    added_ids = {i for s, i in full_vocab.items() if base_vocab.get(s) != i}

    # id -> bytes, skipping added tokens and any char not in the byte alphabet
    id_to_bytes: dict[int, bytes] = {}
    for s, i in base_vocab.items():
        try:
            id_to_bytes[i] = token_to_bytes(s)
        except KeyError:
            continue  # not a byte-level token (shouldn't happen for Qwen)
    vocab_size = tok.get_vocab_size(with_added_tokens=True)
    print(f"=== Tokenizer: {args.tokenizer} ===")
    print(f"vocab size: {vocab_size:,} (analyzing {len(id_to_bytes):,} byte-level tokens, {len(added_ids)} added/special skipped)")

    dsu = DSU(vocab_size)
    tier_stats: dict[str, int] = {"case": 0, "ws_kind": 0, "ws_run": 0}

    for tier, norm in (("case", norm_case), ("ws_kind", norm_ws_kind), ("ws_run", norm_ws_run)):
        groups: dict[bytes, list[int]] = defaultdict(list)
        for i, bs in id_to_bytes.items():
            groups[norm(bs)].append(i)
        for ids in groups.values():
            for other in ids[1:]:
                if dsu.find(ids[0]) != dsu.find(other):
                    tier_stats[tier] += 1
                    dsu.union(ids[0], other)

    # collect components
    comps: dict[int, list[int]] = defaultdict(list)
    for i in id_to_bytes:
        comps[dsu.find(i)].append(i)
    multi = {root: sorted(ids) for root, ids in comps.items() if len(ids) >= args.min_group}

    total_remapped = sum(len(ids) - 1 for ids in multi.values())
    print(f"\n=== Merge candidates ===")
    print(f"new unions by tier: case-only={tier_stats['case']:,}, ws-kind={tier_stats['ws_kind']:,}, ws-run={tier_stats['ws_run']:,}")
    print(f"groups with >= {args.min_group} members: {len(multi):,}")
    print(f"tokens that can be remapped:  {total_remapped:,}")
    print(f"compressed vocab size:        {vocab_size - total_remapped:,}  (was {vocab_size:,}, -{100.0 * total_remapped / vocab_size:.2f}%)")

    # component details; canonical = member whose bytes equal the fully-normalized form if present, else min id
    print(f"\n=== Largest groups (showing up to {args.show_groups}) ===")
    for _, ids in sorted(multi.items(), key=lambda kv: -len(kv[1]))[: args.show_groups]:
        bss = {i: id_to_bytes[i] for i in ids}
        full_norm = norm_ws_run(bss[min(ids)])
        canon = next((i for i in ids if bss[i] == full_norm and bss[i] == norm_ws_run(bss[i])), None)
        if canon is None:
            canon = min(ids)
        members = ", ".join(f"{show(bss[i])}({i})" for i in ids if i != canon)
        print(f"  {show(bss[canon])}({canon})  <-  {members}")

    if args.out:
        mapping: dict[str, int] = {}
        for _, ids in multi.items():
            bss = {i: id_to_bytes[i] for i in ids}
            canon = min(ids)
            full_norm = norm_ws_run(bss[canon])
            better = [i for i in ids if bss[i] == full_norm]
            if better:
                canon = min(better)
            for i in ids:
                if i != canon:
                    mapping[str(i)] = canon
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "tokenizer": str(args.tokenizer),
                    "vocab_size": vocab_size,
                    "groups": len(multi),
                    "remapped": total_remapped,
                    "compressed_vocab_size": vocab_size - total_remapped,
                    "map": mapping,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        print(f"\n>> id->id mapping for {len(mapping):,} tokens written to {args.out}")


if __name__ == "__main__":
    main()