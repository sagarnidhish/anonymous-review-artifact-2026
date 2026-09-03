"""Particle-aware split definitions for the paired-temperature GRA29 movies."""

from __future__ import annotations

import re
from dataclasses import dataclass


_STEM_PATTERN = re.compile(
    r"^GRA29_C20_(?P<temperature>25deg|45deg)_particle(?P<particle>[1-4])$"
)


def particle_identity_from_stem(stem: str) -> int:
    """Return the physical particle number encoded by a canonical GRA29 stem."""
    match = _STEM_PATTERN.fullmatch(stem)
    if match is None:
        raise ValueError(f"not a canonical GRA29 particle stem: {stem}")
    return int(match.group("particle"))


@dataclass(frozen=True)
class ParticleFold:
    """One identity holdout with separate same- and cross-temperature tests."""

    heldout_particle: int
    train_stems: frozenset[str]
    same_temperature_test_stems: frozenset[str]
    cross_temperature_test_stems: frozenset[str]

    @property
    def all_test_stems(self) -> frozenset[str]:
        return self.same_temperature_test_stems | self.cross_temperature_test_stems

    def evaluation_group_for_stem(self, stem: str) -> str:
        if stem in self.same_temperature_test_stems:
            return "same_temperature_unseen_particle"
        if stem in self.cross_temperature_test_stems:
            return "cross_temperature_unseen_particle"
        raise ValueError(f"stem outside identity-holdout test set: {stem}")


def build_identity_holdout_fold(heldout_particle: int) -> ParticleFold:
    """Train on the other three 25 C particles and test the held-out pair."""
    if heldout_particle not in range(1, 5):
        raise ValueError(
            f"heldout_particle must be one of 1, 2, 3, 4; got {heldout_particle}"
        )

    train_stems = frozenset(
        f"GRA29_C20_25deg_particle{particle}"
        for particle in range(1, 5)
        if particle != heldout_particle
    )
    same_temperature = frozenset(
        {f"GRA29_C20_25deg_particle{heldout_particle}"}
    )
    cross_temperature = frozenset(
        {f"GRA29_C20_45deg_particle{heldout_particle}"}
    )
    return ParticleFold(
        heldout_particle=heldout_particle,
        train_stems=train_stems,
        same_temperature_test_stems=same_temperature,
        cross_temperature_test_stems=cross_temperature,
    )
