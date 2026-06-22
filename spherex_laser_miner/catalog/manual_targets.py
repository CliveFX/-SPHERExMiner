from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ManualTarget:
    target_id: str
    target_type: str
    object_name: str
    ra_deg: float
    dec_deg: float
    reference_epoch_yr: float
    pmra_masyr: float | None
    pmdec_masyr: float | None
    parallax_mas: float | None
    source_catalog: str
    source_catalog_id: str
    priority_score: float
    notes: str = ""

    @classmethod
    def from_mapping(cls, row: dict[str, Any]) -> "ManualTarget":
        return cls(
            target_id=str(row["target_id"]),
            target_type=str(row["target_type"]),
            object_name=str(row["object_name"]),
            ra_deg=float(row["ra_deg"]),
            dec_deg=float(row["dec_deg"]),
            reference_epoch_yr=float(row["reference_epoch_yr"]),
            pmra_masyr=_optional_float(row.get("pmra_masyr")),
            pmdec_masyr=_optional_float(row.get("pmdec_masyr")),
            parallax_mas=_optional_float(row.get("parallax_mas")),
            source_catalog=str(row["source_catalog"]),
            source_catalog_id=str(row["source_catalog_id"]),
            priority_score=float(row.get("priority_score", 0.0)),
            notes=str(row.get("notes", "")),
        )


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_manual_targets(path: Path) -> list[ManualTarget]:
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    return [ManualTarget.from_mapping(row) for row in doc.get("targets", [])]


def get_manual_target(path: Path, target_id: str) -> ManualTarget:
    for target in load_manual_targets(path):
        if target.target_id == target_id:
            return target
    raise KeyError(f"Manual target not found: {target_id}")
