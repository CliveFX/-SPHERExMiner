from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from io import BytesIO

import requests
from astropy.table import Table


SIA_URL = "https://irsa.ipac.caltech.edu/SIA"


@dataclass(frozen=True)
class SpherexImageCandidate:
    s_ra: float
    s_dec: float
    instrument_name: str
    detector: int
    obs_id: str
    access_url: str
    access_format: str
    access_estsize_kb: int | None
    s_region: str
    obs_collection: str
    obs_release_date: str
    t_min_mjd: float | None
    t_max_mjd: float | None
    cloud_access: str
    distance_rank: int

    @property
    def obs_mid_mjd(self) -> float | None:
        if self.t_min_mjd is None or self.t_max_mjd is None:
            return None
        return (self.t_min_mjd + self.t_max_mjd) / 2.0

    @property
    def s3_uri(self) -> str | None:
        if not self.cloud_access:
            return None
        try:
            doc = json.loads(self.cloud_access)
            aws = doc.get("aws", {})
            return f"s3://{aws['bucket_name']}/{aws['key']}"
        except Exception:
            return None

    def to_json_dict(self) -> dict[str, object]:
        row = asdict(self)
        row["obs_mid_mjd"] = self.obs_mid_mjd
        row["s3_uri"] = self.s3_uri
        return row


def query_sia_candidates(ra_deg: float, dec_deg: float, radius_deg: float = 0.01) -> list[SpherexImageCandidate]:
    params = {
        "COLLECTION": "spherex_qr2",
        "POS": f"circle {ra_deg} {dec_deg} {radius_deg}",
        "RESPONSEFORMAT": "VOTABLE",
    }
    response = requests.get(SIA_URL, params=params, timeout=120)
    response.raise_for_status()
    table = Table.read(BytesIO(response.content), format="votable")
    return [_candidate_from_sia_row(table, idx) for idx in range(len(table))]


def _candidate_from_sia_row(table: Table, idx: int) -> SpherexImageCandidate:
    row = table[idx]
    instrument = str(row["col_7"])
    detector = int(instrument.rsplit("D", 1)[1]) if "D" in instrument else -1
    return SpherexImageCandidate(
        s_ra=float(row["col_0"]),
        s_dec=float(row["col_1"]),
        instrument_name=instrument,
        detector=detector,
        obs_id=str(row["col_9"]),
        access_url=str(row["col_15"]),
        access_format=str(row["col_16"]),
        access_estsize_kb=_optional_int(row["col_17"]),
        s_region=str(row["col_19"]),
        obs_collection=str(row["col_20"]),
        obs_release_date=str(row["col_34"]),
        t_min_mjd=_optional_float(row["col_39"]),
        t_max_mjd=_optional_float(row["col_40"]),
        cloud_access=str(row["col_48"]),
        distance_rank=idx + 1,
    )


def _optional_float(value: object) -> float | None:
    try:
        if getattr(value, "mask", False):
            return None
        return float(value)
    except Exception:
        return None


def _optional_int(value: object) -> int | None:
    try:
        if getattr(value, "mask", False):
            return None
        return int(value)
    except Exception:
        return None
