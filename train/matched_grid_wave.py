#!/usr/bin/env python3
"""Validate and address payload lists for CSD3 backfill waves."""

from __future__ import annotations

import argparse


MAX_WAVE_PAYLOADS = 12


def parse_payload_ids(value: str) -> tuple[int, ...]:
    pieces = value.split(",") if value else []
    if not pieces or any(not piece.strip() for piece in pieces):
        raise ValueError("payload list must be a non-empty comma-separated list")
    try:
        payload_ids = tuple(int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise ValueError("payload IDs must be integers") from exc
    if len(payload_ids) > MAX_WAVE_PAYLOADS:
        raise ValueError(f"a wave may contain at most {MAX_WAVE_PAYLOADS} payloads")
    if len(set(payload_ids)) != len(payload_ids):
        raise ValueError("payload IDs must be unique")
    if any(payload_id < 0 or payload_id > 23 for payload_id in payload_ids):
        raise ValueError("payload IDs must be in 0..23")
    return payload_ids


def payload_for_rank(value: str, rank: int) -> int | None:
    if rank < 0:
        raise ValueError("rank must be non-negative")
    payload_ids = parse_payload_ids(value)
    return payload_ids[rank] if rank < len(payload_ids) else None


def wave_resources(payload_count: int) -> tuple[int, int]:
    """Return complete four-GPU nodes and task slots for a payload wave."""
    if payload_count < 1 or payload_count > MAX_WAVE_PAYLOADS:
        raise ValueError(f"payload count must be in 1..{MAX_WAVE_PAYLOADS}")
    nodes = (payload_count + 3) // 4
    return nodes, nodes * 4


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-ids", required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--resources", action="store_true")
    args = parser.parse_args()
    try:
        payload_ids = parse_payload_ids(args.payload_ids)
        if args.resources and args.rank is not None:
            raise ValueError("--resources and --rank are mutually exclusive")
        if args.resources:
            nodes, tasks = wave_resources(len(payload_ids))
            print(f"{nodes},{tasks}")
        elif args.rank is None:
            print(",".join(str(value) for value in payload_ids))
        else:
            payload_id = payload_for_rank(args.payload_ids, args.rank)
            print("idle" if payload_id is None else payload_id)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
