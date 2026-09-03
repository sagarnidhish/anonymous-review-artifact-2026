"""Metadata builders shared by training and evaluation-only entry points."""

from __future__ import annotations


def build_result_row(
    *,
    stem: str,
    role: str,
    model_family: str,
    tag: str,
    active_fields,
    context_len: int,
    split: str,
    seed: int,
    prediction_form: str,
    heldout_particle: int | None = None,
    evaluation_group: str | None = None,
) -> dict:
    """Return the metadata contract required by reference artifact savers."""
    row = {
        "stem": stem,
        "role": role,
        "model_family": model_family,
        "tag": tag,
        "active_fields": list(active_fields),
        "context_len": int(context_len),
        "split": split,
        "seed": int(seed),
        "prediction_form": prediction_form,
    }
    if heldout_particle is not None:
        row["heldout_particle"] = int(heldout_particle)
    if evaluation_group is not None:
        row["evaluation_group"] = str(evaluation_group)
    return row


def rows_for_role(rows, role: str) -> list[dict]:
    """Select independent evaluation units for one declared data role."""
    return [row for row in rows if row.get("role") == role]
