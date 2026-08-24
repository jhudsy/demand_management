"""Configuration objects for the demand-management model.

Every run is fully described by a `RunConfig`, which is serialisable and written
alongside its results so any figure can be regenerated from its manifest.

The institutional block is the load-bearing part of this model. Institutions
differ in exactly two things:

    size    -- how many researchers they have
    quality -- how good, on average, those researchers are

Deliberately NOT modelled: why an institution is strong (talent, research office,
reputation). Triage sees a ranking and cannot distinguish the sources, so
separating them adds parameters without changing any decision. Reviewer bias
towards prestigious names is a peer-review problem that exists with or without a
cap, is not caused by demand management, and belongs to a different intervention.
Individual track record is not modelled separately either: a researcher's latent
quality is the only thing that distinguishes them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

SizeDist = Literal["equal", "lognormal", "powerlaw", "empirical"]


@dataclass(frozen=True)
class InstitutionConfig:
    """How institutions differ from one another."""

    n_institutions: int = 50
    n_researchers: int = 2000

    # --- capacity channel ------------------------------------------------
    size_dist: SizeDist = "equal"
    # Spread of the size distribution. For "lognormal" this is sigma of the
    # underlying normal; for "powerlaw" it is the Pareto shape (smaller = more
    # skewed). Ignored when size_dist == "equal".
    size_spread: float = 0.8
    min_size: int = 5

    # --- quality channel ---------------------------------------------------
    # Literally corr(researcher latent quality, institution effect).
    #   0   = quality independent of affiliation. This is the null case AND the
    #         assumption made by the prior ABM in this project's anchor paper,
    #         so it is the case in which the unit of the cap must not matter.
    #   >0  = good researchers concentrate in some institutions.
    rho_quality: float = 0.0

    # Are larger institutions also better? Correlation between log size and the
    # institution quality effect.
    corr_size_quality: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.rho_quality < 1.0:
            raise ValueError("rho_quality must be in [0, 1)")
        if not -1.0 <= self.corr_size_quality <= 1.0:
            raise ValueError("corr_size_quality must be in [-1, 1]")
        if self.n_institutions < 1 or self.n_researchers < self.n_institutions:
            raise ValueError("need at least one researcher per institution")
        if self.min_size < 1:
            raise ValueError("min_size must be >= 1")


@dataclass(frozen=True)
class RunConfig:
    institutions: InstitutionConfig = field(default_factory=InstitutionConfig)
    seed: int = 0
    label: str = "unnamed"

    def validate(self) -> None:
        self.institutions.validate()

    def to_dict(self) -> dict:
        return asdict(self)
