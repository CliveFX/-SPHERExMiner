from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HealpixTile:
    nside: int
    hpx: int
    order: str
    vertices: list[tuple[float, float]]

    @property
    def tile_id(self) -> str:
        return f"hpx_nside{self.nside:04d}_{self.order}_{self.hpx:08d}"

    @property
    def s_region(self) -> str:
        parts = ["POLYGON", "ICRS"]
        for ra_deg, dec_deg in self.closed_vertices:
            parts.append(f"{ra_deg:.10f}")
            parts.append(f"{dec_deg:.10f}")
        return " ".join(parts)

    @property
    def closed_vertices(self) -> list[tuple[float, float]]:
        if not self.vertices:
            return []
        return [*self.vertices, self.vertices[0]]


def healpix_tile(nside: int, hpx: int, *, order: str = "nested") -> HealpixTile:
    if nside <= 0 or nside & (nside - 1):
        raise ValueError("nside must be a positive power of two")
    if order not in {"nested", "ring"}:
        raise ValueError("order must be 'nested' or 'ring'")
    max_hpx = 12 * nside * nside
    if hpx < 0 or hpx >= max_hpx:
        raise ValueError(f"hpx must be in [0, {max_hpx}) for nside={nside}")

    from astropy_healpix import HEALPix

    hp = HEALPix(nside=nside, order=order, frame=None)
    lon, lat = hp.boundaries_lonlat([hpx], step=1)
    vertices = [(float(ra.deg) % 360.0, float(dec.deg)) for ra, dec in zip(lon[0], lat[0])]
    return HealpixTile(nside=nside, hpx=hpx, order=order, vertices=vertices)


def iter_hpx(start: int, count: int) -> list[int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    return [start + idx for idx in range(count)]
