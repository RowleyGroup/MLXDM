#!/usr/bin/env python3
"""
transit_isochrone_map.py
=========================

Generates a colour-coded, interactive isochrone map of OC Transpo (Ottawa-Gatineau)
public-transit travel times to a user-specified destination address, in the style of a
real-estate "commute time" map.

Pipeline
--------
1. Geocode the destination address (Google Geocoding API).
2. Build a lat/lon sample grid covering the Ottawa-Gatineau National Capital Region
   (Ottawa, Gatineau, Kanata, Orleans, Barrhaven, Stittsville, Nepean, Aylmer).
3. Query transit travel time from every grid point to the destination (Google Distance
   Matrix API, transit mode), batching up to 25 origins per request, run concurrently,
   rate-limited, and cached in a local SQLite database so an interrupted run can resume
   without re-spending API quota.
4. Write the raw samples to CSV and GeoJSON.
5. Interpolate the sparse samples onto a fine mesh and extract filled isochrone-band
   polygons (0-15, 15-30, 30-45, 45-60, 60-75, 75-90, >90 minutes) using Matplotlib's
   contouring engine as a pure geometry tool (no plot is ever shown/rendered to screen).
6. Render an interactive Folium/Leaflet HTML map with coloured points, isochrone bands,
   a legend, hover tooltips, an optional heatmap layer, and (optionally) a second
   weekday/weekend comparison layer.

Example usage
-------------
    export GOOGLE_MAPS_API_KEY="your-api-key-here"
    python transit_isochrone_map.py \\
        --destination "1125 Colonel By Dr Ottawa ON" \\
        --departure "2026-09-01 08:00" \\
        --grid-km 1.0 \\
        --max-minutes 90

See the accompanying README.md for full Google Cloud API setup instructions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import folium
import googlemaps
import matplotlib

matplotlib.use("Agg")  # Never open a GUI window / display; we only use Matplotlib as a
# geometry engine to turn interpolated travel-time surfaces into isochrone polygons.
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from folium.plugins import HeatMap  # noqa: E402
from googlemaps import exceptions as gmaps_exceptions  # noqa: E402
from scipy.interpolate import griddata  # noqa: E402
from shapely.geometry import Polygon as ShapelyPolygon  # noqa: E402
from shapely.geometry import mapping as shapely_mapping  # noqa: E402
from shapely.ops import unary_union  # noqa: E402
from tqdm import tqdm  # noqa: E402

# --------------------------------------------------------------------------------------
# Configurable constants
# --------------------------------------------------------------------------------------

LOCAL_TIMEZONE = ZoneInfo("America/Toronto")

#: Bounding box comfortably covering all of the requested Ottawa-Gatineau sub-regions.
#: Verified in `validate_region_coverage()` against approximate centre points below.
DEFAULT_MIN_LAT = 45.15
DEFAULT_MAX_LAT = 45.55
DEFAULT_MIN_LON = -76.05
DEFAULT_MAX_LON = -75.42

#: Approximate centre points of the regions the grid is required to cover, used only to
#: sanity-check that the configured bounding box actually contains them.
REGION_REFERENCE_POINTS: dict[str, tuple[float, float]] = {
    "Ottawa (downtown)": (45.4215, -75.6972),
    "Gatineau": (45.4770, -75.7013),
    "Kanata": (45.3088, -75.9188),
    "Orleans": (45.4676, -75.5305),
    "Barrhaven": (45.2733, -75.7466),
    "Stittsville": (45.2544, -75.9161),
    "Nepean": (45.3475, -75.7738),
    "Aylmer": (45.3947, -75.8394),
}

#: Google Distance Matrix API allows at most 25 origins (or 25 destinations) per
#: server-side request. We always keep the "fixed" endpoint (destination, or origin in
#: --reverse mode) at 1 and batch the grid points up to this limit.
MAX_BATCH_SIZE = 25

#: Fixed colour bands per the required real-estate-style legend. Each tuple is
#: (lower_minutes_inclusive, upper_minutes_exclusive, hex_colour, label).
BAND_DEFS: list[tuple[float, float, str, str]] = [
    (0, 15, "#006400", "0-15 min"),      # dark green
    (15, 30, "#90EE90", "15-30 min"),    # light green
    (30, 45, "#FFFF00", "30-45 min"),    # yellow
    (45, 60, "#FFA500", "45-60 min"),    # orange
    (60, 75, "#FF0000", "60-75 min"),    # red
    (75, 90, "#8B0000", "75-90 min"),    # dark red
    (90, math.inf, "#800080", ">90 min"),  # purple
]
UNREACHABLE_COLOR = "#808080"  # gray: no transit route found / query failed

#: Isochrone mesh resolution (points along each axis of the interpolated surface).
ISO_MESH_LAT_POINTS = 180
ISO_MESH_LON_POINTS = 220

#: Simplification tolerance (degrees) applied to isochrone polygons before writing them
#: into the map, to keep the resulting HTML file a reasonable size.
ISO_SIMPLIFY_TOLERANCE_DEG = 0.0015

DEFAULT_CACHE_DB_NAME = "transit_cache.sqlite3"

logger = logging.getLogger("transit_map")


# --------------------------------------------------------------------------------------
# Small data types
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float


@dataclass(frozen=True)
class Bounds:
    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float


@dataclass(frozen=True)
class DestinationInfo:
    lat: float
    lon: float
    formatted_address: str


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------

def setup_logging(output_dir: Path, level: str) -> None:
    """Configure console + file logging. Called once at start-up."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    root.addHandler(console_handler)

    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


# --------------------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------------------

class RateLimiter:
    """A simple thread-safe pacer that enforces a maximum request rate across all
    worker threads. Not a burst-tolerant token bucket by design: OC Transpo/Google
    quota errors are expensive to recover from, so we deliberately smooth requests
    out to a steady cadence instead of allowing bursts.
    """

    def __init__(self, requests_per_second: float) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
        self._next_allowed_time = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed_time - now
            if wait > 0:
                start = self._next_allowed_time
            else:
                start = now
            self._next_allowed_time = start + self._min_interval
        if wait > 0:
            time.sleep(wait)


# --------------------------------------------------------------------------------------
# SQLite cache
# --------------------------------------------------------------------------------------

class CacheDB:
    """Local SQLite cache for geocoding results and per-point travel-time queries.

    This is what makes the tool "resumable": every completed batch is committed to
    disk immediately, so if the process is interrupted (Ctrl-C, crash, network outage)
    re-running the exact same command line skips every point that was already
    successfully queried.
    """

    def __init__(self, path: Path) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS geocode_cache (
                    address TEXT PRIMARY KEY,
                    lat REAL NOT NULL,
                    lon REAL NOT NULL,
                    formatted_address TEXT,
                    cached_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS travel_time_cache (
                    query_hash TEXT PRIMARY KEY,
                    origin_lat REAL NOT NULL,
                    origin_lon REAL NOT NULL,
                    dest_lat REAL NOT NULL,
                    dest_lon REAL NOT NULL,
                    departure_epoch INTEGER NOT NULL,
                    reverse INTEGER NOT NULL,
                    transit_modes TEXT NOT NULL,
                    travel_time_minutes REAL,
                    status TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # -- geocoding -----------------------------------------------------------------

    def get_geocode(self, address: str) -> tuple[float, float, str] | None:
        with self._lock:
            cur = self._conn.execute(
                "SELECT lat, lon, formatted_address FROM geocode_cache WHERE address = ?",
                (address,),
            )
            row = cur.fetchone()
        return (row[0], row[1], row[2]) if row else None

    def set_geocode(self, address: str, lat: float, lon: float, formatted_address: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO geocode_cache (address, lat, lon, formatted_address, cached_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (address, lat, lon, formatted_address, datetime.utcnow().isoformat()),
            )
            self._conn.commit()

    # -- travel times ----------------------------------------------------------------

    def get_travel_times_many(self, keys: list[str]) -> dict[str, tuple[float | None, str]]:
        results: dict[str, tuple[float | None, str]] = {}
        chunk_size = 900  # keep well under SQLite's default 999 bound-parameter limit
        with self._lock:
            for i in range(0, len(keys), chunk_size):
                chunk = keys[i : i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                cur = self._conn.execute(
                    f"SELECT query_hash, travel_time_minutes, status FROM travel_time_cache "
                    f"WHERE query_hash IN ({placeholders})",
                    chunk,
                )
                for query_hash, minutes, status in cur.fetchall():
                    results[query_hash] = (minutes, status)
        return results

    def set_travel_times_many(self, records: list[tuple]) -> None:
        """records: list of (query_hash, origin_lat, origin_lon, dest_lat, dest_lon,
        departure_epoch, reverse, transit_modes, travel_time_minutes, status)."""
        if not records:
            return
        now = datetime.utcnow().isoformat()
        rows = [r + (now,) for r in records]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO travel_time_cache "
                "(query_hash, origin_lat, origin_lon, dest_lat, dest_lon, departure_epoch, "
                " reverse, transit_modes, travel_time_minutes, status, cached_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


# --------------------------------------------------------------------------------------
# Geocoding
# --------------------------------------------------------------------------------------

def call_with_retry(
    func,
    *args,
    max_retries: int = 5,
    base_delay: float = 1.5,
    **kwargs,
):
    """Call a googlemaps client method, retrying transient failures with exponential
    backoff + jitter. Non-transient errors (bad API key, API not enabled, malformed
    request) are re-raised immediately since retrying cannot fix them.
    """
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except (gmaps_exceptions.Timeout, gmaps_exceptions.TransportError) as exc:
            attempt += 1
            if attempt > max_retries:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Transient network error (%s); retrying in %.1fs (attempt %d/%d)",
                exc, delay, attempt, max_retries,
            )
            time.sleep(delay)
        except gmaps_exceptions.ApiError as exc:
            status = getattr(exc, "status", "")
            if status == "OVER_QUERY_LIMIT":
                attempt += 1
                if attempt > max_retries:
                    raise
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.warning(
                    "Over query limit; backing off %.1fs (attempt %d/%d)",
                    delay, attempt, max_retries,
                )
                time.sleep(delay)
            else:
                # REQUEST_DENIED, INVALID_REQUEST, etc: not recoverable by retrying.
                raise


def geocode_destination(gmaps: googlemaps.Client, cache: CacheDB, address: str) -> DestinationInfo:
    """Resolve a free-text address to (lat, lon) via the Google Geocoding API,
    using the local cache to avoid re-geocoding the same address on every run."""
    cached = cache.get_geocode(address)
    if cached is not None:
        lat, lon, formatted = cached
        logger.info("Using cached geocode for %r -> %s (%.5f, %.5f)", address, formatted, lat, lon)
        return DestinationInfo(lat=lat, lon=lon, formatted_address=formatted)

    logger.info("Geocoding address: %r", address)
    try:
        results = call_with_retry(gmaps.geocode, address)
    except Exception as exc:  # noqa: BLE001 - surfaced to the user as a fatal error
        raise RuntimeError(f"Failed to geocode destination address {address!r}: {exc}") from exc

    if not results:
        raise RuntimeError(
            f"Google Geocoding API returned no results for address: {address!r}. "
            "Check the spelling/format of the address."
        )

    top = results[0]
    location = top["geometry"]["location"]
    formatted_address = top.get("formatted_address", address)
    cache.set_geocode(address, location["lat"], location["lng"], formatted_address)
    return DestinationInfo(lat=location["lat"], lon=location["lng"], formatted_address=formatted_address)


# --------------------------------------------------------------------------------------
# Grid generation
# --------------------------------------------------------------------------------------

def parse_region_bounds(spec: str) -> Bounds:
    """Parse a 'min_lat,min_lon,max_lat,max_lon' override string."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError("--region-bounds must be 'min_lat,min_lon,max_lat,max_lon'")
    min_lat, min_lon, max_lat, max_lon = (float(p) for p in parts)
    if min_lat >= max_lat or min_lon >= max_lon:
        raise ValueError("--region-bounds: min values must be less than max values")
    return Bounds(min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon)


def validate_region_coverage(bounds: Bounds) -> None:
    """Warn (but do not fail) if the configured bounding box does not actually cover
    one of the required sub-regions, so misconfiguration via --region-bounds is caught
    early rather than silently producing an incomplete map."""
    for name, (lat, lon) in REGION_REFERENCE_POINTS.items():
        inside = bounds.min_lat <= lat <= bounds.max_lat and bounds.min_lon <= lon <= bounds.max_lon
        if not inside:
            logger.warning(
                "Region %r reference point (%.4f, %.4f) falls OUTSIDE the configured grid "
                "bounds (%.4f..%.4f lat, %.4f..%.4f lon) - it will not be sampled.",
                name, lat, lon, bounds.min_lat, bounds.max_lat, bounds.min_lon, bounds.max_lon,
            )


def generate_grid(bounds: Bounds, grid_km: float) -> list[GeoPoint]:
    """Build a regular lat/lon sample grid across the bounding box at approximately
    `grid_km` kilometre spacing (converted from great-circle distance using the
    bounding box's mid-latitude)."""
    if grid_km <= 0:
        raise ValueError("--grid-km must be positive")

    mid_lat = (bounds.min_lat + bounds.max_lat) / 2.0
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(math.radians(mid_lat))

    lat_step = grid_km / km_per_deg_lat
    lon_step = grid_km / km_per_deg_lon

    lats = np.arange(bounds.min_lat, bounds.max_lat + lat_step / 2.0, lat_step)
    lons = np.arange(bounds.min_lon, bounds.max_lon + lon_step / 2.0, lon_step)

    return [GeoPoint(lat=float(la), lon=float(lo)) for la in lats for lo in lons]


# --------------------------------------------------------------------------------------
# Travel time queries
# --------------------------------------------------------------------------------------

def make_query_key(
    origin: GeoPoint,
    dest: GeoPoint,
    departure_epoch: int,
    reverse: bool,
    transit_modes: list[str] | None,
) -> str:
    """Deterministic cache key covering every parameter that affects the result, so
    changing --departure, --reverse, or --transit-modes never returns a stale answer."""
    payload = (
        f"{origin.lat:.6f},{origin.lon:.6f}|{dest.lat:.6f},{dest.lon:.6f}|"
        f"{departure_epoch}|{int(reverse)}|{','.join(sorted(transit_modes or []))}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def query_batch(
    gmaps: googlemaps.Client,
    rate_limiter: RateLimiter,
    origins: list[GeoPoint],
    destinations: list[GeoPoint],
    departure_epoch: int,
    transit_modes: list[str] | None,
) -> dict:
    """Issue a single Distance Matrix API request for up to MAX_BATCH_SIZE
    origin/destination pairs (one side of the pair is always length 1)."""
    rate_limiter.acquire()
    kwargs = dict(
        origins=[f"{p.lat:.6f},{p.lon:.6f}" for p in origins],
        destinations=[f"{p.lat:.6f},{p.lon:.6f}" for p in destinations],
        mode="transit",
        departure_time=departure_epoch,
        units="metric",
    )
    if transit_modes:
        kwargs["transit_mode"] = transit_modes
    return call_with_retry(gmaps.distance_matrix, **kwargs)


def run_travel_time_queries(
    gmaps: googlemaps.Client,
    cache: CacheDB,
    rate_limiter: RateLimiter,
    grid_points: list[GeoPoint],
    destination: DestinationInfo,
    departure_epoch: int,
    reverse: bool,
    transit_modes: list[str] | None,
    max_workers: int,
    batch_size: int,
    force_refresh: bool,
) -> pd.DataFrame:
    """Query (or fetch from cache) the transit travel time between every grid point and
    the destination, concurrently and rate-limited, persisting each completed batch to
    the SQLite cache immediately so the run is resumable after an interruption.

    In normal (non-reverse) mode, grid points are the *origins* and the destination
    address is the fixed *destination* -- i.e. "how long to commute TO the
    destination from here". With --reverse, the roles are swapped: the destination
    address becomes the fixed *origin* and travel time is computed FROM it TO each
    grid point -- useful for treating the specified address as a starting point
    (e.g. "if I live here, how far can transit take me").
    """
    dest_point = GeoPoint(destination.lat, destination.lon)
    keys = [
        make_query_key(gp, dest_point, departure_epoch, reverse, transit_modes)
        for gp in grid_points
    ]

    cached_results: dict[str, tuple[float | None, str]] = {}
    if not force_refresh:
        cached_results = cache.get_travel_times_many(keys)
        if cached_results:
            logger.info(
                "Resume: %d/%d grid points already cached for this exact query",
                len(cached_results), len(grid_points),
            )

    pending_idx = [i for i, k in enumerate(keys) if k not in cached_results]
    batch_size = min(batch_size, MAX_BATCH_SIZE)
    batches = [pending_idx[i : i + batch_size] for i in range(0, len(pending_idx), batch_size)]
    logger.info(
        "Querying %d grid points in %d batches (batch size %d, %d workers)",
        len(pending_idx), len(batches), batch_size, max_workers,
    )

    results: dict[str, tuple[float | None, str]] = dict(cached_results)

    def process_batch(idx_batch: list[int]) -> list[tuple]:
        batch_points = [grid_points[i] for i in idx_batch]
        origins = batch_points if not reverse else [dest_point]
        destinations = [dest_point] if not reverse else batch_points

        try:
            response = query_batch(gmaps, rate_limiter, origins, destinations, departure_epoch, transit_modes)
        except Exception:
            logger.exception("Batch of %d points failed permanently; marking as errored", len(idx_batch))
            return [
                (
                    keys[i], gp.lat, gp.lon, destination.lat, destination.lon,
                    departure_epoch, int(reverse), ",".join(transit_modes or []),
                    None, "REQUEST_ERROR",
                )
                for i, gp in zip(idx_batch, batch_points)
            ]

        rows = response.get("rows", [])
        records = []
        for pos, i in enumerate(idx_batch):
            gp = batch_points[pos]
            element = None
            if not reverse:
                if pos < len(rows) and rows[pos].get("elements"):
                    element = rows[pos]["elements"][0]
            else:
                if rows and pos < len(rows[0].get("elements", [])):
                    element = rows[0]["elements"][pos]

            status = element.get("status", "UNKNOWN") if element else "NO_RESPONSE"
            minutes = None
            if element is not None and status == "OK":
                minutes = element["duration"]["value"] / 60.0

            records.append((
                keys[i], gp.lat, gp.lon, destination.lat, destination.lon,
                departure_epoch, int(reverse), ",".join(transit_modes or []),
                minutes, status,
            ))
        return records

    if batches:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_batch, b): b for b in batches}
            with tqdm(total=len(pending_idx), desc="Querying transit times", unit="pt") as pbar:
                for future in as_completed(futures):
                    records = future.result()
                    cache.set_travel_times_many(records)  # persist immediately -> resumable
                    for rec in records:
                        results[rec[0]] = (rec[8], rec[9])
                    pbar.update(len(records))

    rows_out = []
    for gp, key in zip(grid_points, keys):
        minutes, status = results.get(key, (None, "MISSING"))
        rows_out.append({"lat": gp.lat, "lon": gp.lon, "travel_time_minutes": minutes, "status": status})
    return pd.DataFrame(rows_out)


# --------------------------------------------------------------------------------------
# Output files: CSV / GeoJSON
# --------------------------------------------------------------------------------------

def write_csv(df: pd.DataFrame, path: Path) -> None:
    df[["lat", "lon", "travel_time_minutes", "status"]].rename(
        columns={"lat": "latitude", "lon": "longitude"}
    ).to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))


def build_geojson(df: pd.DataFrame) -> dict:
    features = []
    for row in df.itertuples():
        minutes = None if pd.isna(row.travel_time_minutes) else float(row.travel_time_minutes)
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row.lon, row.lat]},
            "properties": {
                "travel_time_minutes": minutes,
                "status": row.status,
            },
        })
    return {"type": "FeatureCollection", "features": features}


def write_geojson(geojson_obj: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(geojson_obj, f)
    logger.info("Wrote %s (%d features)", path, len(geojson_obj["features"]))


# --------------------------------------------------------------------------------------
# Isochrone polygon computation
# --------------------------------------------------------------------------------------

def interpolate_surface(
    df: pd.DataFrame, bounds: Bounds
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """Interpolate the sparse (lat, lon, minutes) samples onto a regular fine mesh
    covering `bounds`, using linear interpolation inside the sample convex hull and
    nearest-neighbour extrapolation outside it (so the whole bounding box is filled,
    including areas beyond the outermost sampled points)."""
    valid = df["travel_time_minutes"].notna()
    if valid.sum() == 0:
        logger.warning("No successful travel-time queries; skipping isochrone computation")
        return None, None, None

    lats = df.loc[valid, "lat"].to_numpy()
    lons = df.loc[valid, "lon"].to_numpy()
    times = df.loc[valid, "travel_time_minutes"].to_numpy()

    fine_lat = np.linspace(bounds.min_lat, bounds.max_lat, ISO_MESH_LAT_POINTS)
    fine_lon = np.linspace(bounds.min_lon, bounds.max_lon, ISO_MESH_LON_POINTS)
    grid_lon, grid_lat = np.meshgrid(fine_lon, fine_lat)

    nearest = griddata((lons, lats), times, (grid_lon, grid_lat), method="nearest")
    try:
        linear = griddata((lons, lats), times, (grid_lon, grid_lat), method="linear")
        nan_mask = np.isnan(linear)
        linear[nan_mask] = nearest[nan_mask]
        surface = linear
    except Exception:
        logger.warning("Linear interpolation failed (too few/degenerate points); using nearest-neighbour only")
        surface = nearest

    return fine_lon, fine_lat, surface


def _polygons_only(geom):
    """Strip any degenerate non-polygonal pieces (stray points/lines) that can appear
    after a unary_union of near-touching or self-intersecting rings."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    if geom.geom_type == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if not polys:
            return None
        return unary_union(polys)
    return None


def compute_isochrone_bands(
    fine_lon: np.ndarray, fine_lat: np.ndarray, surface: np.ndarray, levels: list[float]
) -> list[tuple[float, float, object]]:
    """Turn an interpolated travel-time surface into filled isochrone-band polygons.

    Uses Matplotlib's contour engine purely as a geometry engine (a Figure is created
    and immediately discarded -- nothing is ever displayed or saved as an image) to
    extract polygon rings for each [levels[i], levels[i+1]) band, then repairs and
    merges them with Shapely. Exterior/hole winding is not distinguished; for a
    single-destination commute-time surface this has no meaningful visual effect and
    keeps the geometry code simple and robust.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    try:
        cs = ax.contourf(fine_lon, fine_lat, surface, levels=levels)
        bands: list[tuple[float, float, object]] = []
        for i in range(len(levels) - 1):
            rings = cs.allsegs[i]
            polygons = []
            for ring in rings:
                if len(ring) < 3:
                    continue
                poly = ShapelyPolygon(ring)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    polygons.append(poly)
            if not polygons:
                continue
            merged = unary_union(polygons)
            merged = _polygons_only(merged.buffer(0))
            if merged is None or merged.is_empty:
                continue
            merged = merged.simplify(ISO_SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
            if not merged.is_empty:
                bands.append((levels[i], levels[i + 1], merged))
        return bands
    finally:
        plt.close(fig)


# --------------------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------------------

def get_band_color(minutes: float | None) -> str:
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return UNREACHABLE_COLOR
    for lower, upper, color, _label in BAND_DEFS:
        if lower <= minutes < upper:
            return color
    return BAND_DEFS[-1][2]


def get_continuous_color(minutes: float | None, max_minutes: float, cmap_name: str = "RdYlGn_r") -> str:
    if minutes is None or (isinstance(minutes, float) and math.isnan(minutes)):
        return UNREACHABLE_COLOR
    cmap = matplotlib.colormaps[cmap_name]
    frac = max(0.0, min(1.0, minutes / max_minutes))
    r, g, b, _a = cmap(frac)
    return matplotlib.colors.to_hex((r, g, b))


def point_color(minutes: float | None, args: argparse.Namespace) -> str:
    if args.color_scheme == "continuous":
        return get_continuous_color(minutes, args.max_minutes)
    return get_band_color(minutes)


# --------------------------------------------------------------------------------------
# Map building
# --------------------------------------------------------------------------------------

def _add_isochrone_layer(m: folium.Map, bands, layer_name: str, dash: bool, show: bool) -> None:
    group = folium.FeatureGroup(name=layer_name, show=show)
    for lower, upper, geom in bands:
        color = get_band_color(lower + 1e-6)
        label = f"{lower:.0f}-{upper:.0f} min" if math.isfinite(upper) else f">{lower:.0f} min"
        feature = {
            "type": "Feature",
            "geometry": shapely_mapping(geom),
            "properties": {"band_label": label},
        }
        style = {
            "fillColor": color,
            "color": color,
            "weight": 1.5,
            "fillOpacity": 0.32,
            "opacity": 0.65,
        }
        if dash:
            style["dashArray"] = "6,4"
        folium.GeoJson(
            feature,
            style_function=lambda _f, style=style: style,
            tooltip=folium.GeoJsonTooltip(fields=["band_label"], aliases=["Travel time band:"]),
        ).add_to(group)
    group.add_to(m)


def _add_points_layer(m: folium.Map, df: pd.DataFrame, args: argparse.Namespace, layer_name: str, show: bool) -> None:
    group = folium.FeatureGroup(name=layer_name, show=show)
    for row in df.itertuples():
        minutes = None if pd.isna(row.travel_time_minutes) else float(row.travel_time_minutes)
        color = point_color(minutes, args)
        label = "No transit route found" if minutes is None else f"{minutes:.0f} min"
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=3,
            color=color,
            weight=0.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(label),
        ).add_to(group)
    group.add_to(m)


def build_legend_html() -> str:
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="background:{color};width:14px;height:14px;display:inline-block;'
        f'margin-right:6px;border:1px solid #333;"></span>{label}</div>'
        for _lower, _upper, color, label in BAND_DEFS
    )
    rows += (
        '<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="background:{UNREACHABLE_COLOR};width:14px;height:14px;display:inline-block;'
        'margin-right:6px;border:1px solid #333;"></span>No transit route found</div>'
    )
    return f"""
    <div style="
        position: fixed; bottom: 20px; left: 20px; z-index: 9999;
        background: white; padding: 10px 12px; border: 2px solid #444;
        border-radius: 6px; font-size: 13px; font-family: Arial, Helvetica, sans-serif;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.35);">
      <div style="font-weight:bold;margin-bottom:4px;">Transit travel time</div>
      {rows}
    </div>
    """


def build_title_html(destination: DestinationInfo, departure_label: str, reverse: bool) -> str:
    verb = "from" if reverse else "to"
    return f"""
    <div style="
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%); z-index: 9999;
        background: white; padding: 8px 16px; border: 2px solid #444; border-radius: 6px;
        font-family: Arial, Helvetica, sans-serif; text-align: center;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.35);">
      <div style="font-weight:bold;font-size:16px;">
        OC Transpo transit travel time {verb} {destination.formatted_address}
      </div>
      <div style="font-size:12px;color:#333;">Departure: {departure_label}</div>
    </div>
    """


def build_map(
    df: pd.DataFrame,
    bands: list[tuple[float, float, object]],
    destination: DestinationInfo,
    departure_label: str,
    args: argparse.Namespace,
    comparison_df: pd.DataFrame | None = None,
    comparison_bands: list[tuple[float, float, object]] | None = None,
    comparison_label: str | None = None,
) -> folium.Map:
    m = folium.Map(location=[destination.lat, destination.lon], zoom_start=11, tiles="CartoDB positron", control_scale=True)

    if bands:
        _add_isochrone_layer(m, bands, "Isochrone bands (primary)", dash=False, show=True)
    _add_points_layer(m, df, args, "Sample points (primary)", show=True)

    if comparison_df is not None:
        if comparison_bands:
            _add_isochrone_layer(m, comparison_bands, f"Isochrone bands ({comparison_label})", dash=True, show=False)
        _add_points_layer(m, comparison_df, args, f"Sample points ({comparison_label})", show=False)

    if args.heatmap:
        heat_points = [
            [row.lat, row.lon, max(0.0, args.max_minutes - min(float(row.travel_time_minutes), args.max_minutes))]
            for row in df.itertuples()
            if not pd.isna(row.travel_time_minutes)
        ]
        if heat_points:
            HeatMap(
                heat_points,
                name="Proximity heatmap (hotter = shorter commute)",
                radius=18,
                blur=22,
                show=False,
            ).add_to(m)

    icon_kwargs = dict(color="blue", icon="star", prefix="fa")
    folium.Marker(
        location=[destination.lat, destination.lon],
        tooltip=destination.formatted_address,
        icon=folium.Icon(**icon_kwargs),
    ).add_to(m)

    m.get_root().html.add_child(folium.Element(build_title_html(destination, departure_label, args.reverse)))
    m.get_root().html.add_child(folium.Element(build_legend_html()))
    folium.LayerControl(collapsed=False).add_to(m)
    return m


# --------------------------------------------------------------------------------------
# Date/time helpers
# --------------------------------------------------------------------------------------

def parse_datetime_arg(value: str) -> datetime:
    """Parse '--departure' as a naive local (America/Toronto) date/time and attach the
    timezone. Google's transit routing requires a departure time in the future."""
    try:
        naive = datetime.strptime(value, "%Y-%m-%d %H:%M")
    except ValueError as exc:
        raise ValueError(f"--departure must be in 'YYYY-MM-DD HH:MM' format, got {value!r}") from exc
    aware = naive.replace(tzinfo=LOCAL_TIMEZONE)
    now = datetime.now(tz=LOCAL_TIMEZONE)
    if aware < now - timedelta(minutes=1):
        raise ValueError(
            f"--departure {value!r} is in the past ({aware.isoformat()}); Google's transit "
            "routing API requires a departure time now or in the future."
        )
    return aware


def compute_comparison_datetime(departure_dt: datetime) -> datetime:
    """Return the same time-of-day on the nearest day in the *other* category
    (weekday vs. weekend) relative to `departure_dt`, for --weekend-compare."""
    is_weekend = departure_dt.weekday() >= 5  # Saturday=5, Sunday=6
    if is_weekend:
        days_ahead = (7 - departure_dt.weekday()) % 7  # next Monday
        days_ahead = days_ahead or 7
    else:
        days_ahead = (5 - departure_dt.weekday()) % 7  # next Saturday
        days_ahead = days_ahead or 7
    return departure_dt + timedelta(days=days_ahead)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a colour-coded interactive isochrone map of OC Transpo "
        "public-transit travel times to (or from) a destination address across "
        "Ottawa-Gatineau.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            '  export GOOGLE_MAPS_API_KEY="your-api-key-here"\n'
            "  python transit_isochrone_map.py \\\n"
            '      --destination "1125 Colonel By Dr Ottawa ON" \\\n'
            '      --departure "2026-09-01 08:00" \\\n'
            "      --grid-km 1.0 \\\n"
            "      --max-minutes 90\n"
        ),
    )
    parser.add_argument("--destination", required=True, help="Destination street address, e.g. '1125 Colonel By Dr Ottawa ON'")
    parser.add_argument("--departure", required=True, help="Departure date/time, local Ottawa time, format 'YYYY-MM-DD HH:MM'")
    parser.add_argument("--grid-km", type=float, default=1.0, help="Sample grid spacing in kilometres (default: 1.0)")
    parser.add_argument("--max-minutes", type=int, default=90, help="Travel-time cap used for the colour scale/heatmap (default: 90)")
    parser.add_argument("--output-dir", default="./output", help="Directory to write CSV/GeoJSON/HTML outputs into (default: ./output)")
    parser.add_argument("--api-key", default=None, help="Google Maps API key (falls back to GOOGLE_MAPS_API_KEY env var)")
    parser.add_argument("--cache-db", default=None, help="Path to the SQLite cache DB (default: <output-dir>/transit_cache.sqlite3)")
    parser.add_argument("--max-workers", type=int, default=4, help="Number of concurrent API request threads (default: 4)")
    parser.add_argument("--requests-per-second", type=float, default=8.0, help="Max Distance Matrix API requests/second across all threads (default: 8.0)")
    parser.add_argument("--batch-size", type=int, default=MAX_BATCH_SIZE, help=f"Grid points per Distance Matrix request, max {MAX_BATCH_SIZE} (default: {MAX_BATCH_SIZE})")
    parser.add_argument("--transit-modes", default=None, help="Comma-separated subset of bus,subway,train,tram,rail to restrict transit routing to (default: any)")
    parser.add_argument("--reverse", action="store_true", help="Compute travel time FROM the destination address TO each grid point, instead of TO the destination")
    parser.add_argument("--heatmap", action="store_true", help="Add an optional toggleable proximity heatmap layer")
    parser.add_argument("--color-scheme", choices=["bands", "continuous"], default="bands", help="Point colouring: fixed 15-minute bands, or a continuous gradient (default: bands)")
    parser.add_argument("--weekend-compare", action="store_true", help="Also compute travel times for the same time-of-day on the nearest day of the opposite weekday/weekend category, as a toggleable comparison layer")
    parser.add_argument("--region-bounds", default=None, help="Override the sample grid bounding box: 'min_lat,min_lon,max_lat,max_lon'")
    parser.add_argument("--no-resume", action="store_true", help="Ignore the cache and re-query every grid point from scratch")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity (default: INFO)")

    args = parser.parse_args(argv)

    if args.grid_km <= 0:
        parser.error("--grid-km must be positive")
    if args.max_minutes <= 0:
        parser.error("--max-minutes must be positive")
    if args.batch_size > MAX_BATCH_SIZE:
        print(
            f"WARNING: --batch-size {args.batch_size} exceeds the Google Distance Matrix API "
            f"limit of {MAX_BATCH_SIZE}; clamping to {MAX_BATCH_SIZE}.",
            file=sys.stderr,
        )
        args.batch_size = MAX_BATCH_SIZE

    return args


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    setup_logging(output_dir, args.log_level)

    logger.info("=== OC Transpo transit isochrone map ===")

    api_key = args.api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not api_key:
        logger.error(
            "No Google Maps API key provided. Pass --api-key or set the GOOGLE_MAPS_API_KEY "
            "environment variable. See README.md for setup instructions."
        )
        return 1

    try:
        departure_dt = parse_datetime_arg(args.departure)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    bounds = (
        parse_region_bounds(args.region_bounds)
        if args.region_bounds
        else Bounds(DEFAULT_MIN_LAT, DEFAULT_MAX_LAT, DEFAULT_MIN_LON, DEFAULT_MAX_LON)
    )
    validate_region_coverage(bounds)

    cache_db_path = Path(args.cache_db) if args.cache_db else output_dir / DEFAULT_CACHE_DB_NAME
    cache_db_path.parent.mkdir(parents=True, exist_ok=True)
    cache = CacheDB(cache_db_path)
    rate_limiter = RateLimiter(args.requests_per_second)
    gmaps = googlemaps.Client(key=api_key, timeout=15)

    transit_modes = [m.strip() for m in args.transit_modes.split(",")] if args.transit_modes else None

    try:
        destination = geocode_destination(gmaps, cache, args.destination)
    except Exception as exc:  # noqa: BLE001
        logger.error(str(exc))
        cache.close()
        return 1

    logger.info(
        "Resolved destination: %s (%.5f, %.5f)",
        destination.formatted_address, destination.lat, destination.lon,
    )

    grid_points = generate_grid(bounds, args.grid_km)
    logger.info("Generated %d grid points at ~%.2f km spacing", len(grid_points), args.grid_km)

    try:
        df = run_travel_time_queries(
            gmaps, cache, rate_limiter, grid_points, destination,
            int(departure_dt.timestamp()), args.reverse, transit_modes,
            args.max_workers, args.batch_size, args.no_resume,
        )
    except KeyboardInterrupt:
        logger.warning(
            "Interrupted. Partial results have already been saved to the cache "
            "(%s) - re-run the same command to resume.", cache_db_path,
        )
        cache.close()
        return 130

    n_ok = df["travel_time_minutes"].notna().sum()
    logger.info("Travel-time query results: %d/%d points reachable", n_ok, len(df))

    write_csv(df, output_dir / "travel_times.csv")
    write_geojson(build_geojson(df), output_dir / "travel_times.geojson")

    fine_lon, fine_lat, surface = interpolate_surface(df, bounds)
    levels = sorted(set([0, 15, 30, 45, 60, 75, 90, max(120, args.max_minutes + 30)]))
    bands = compute_isochrone_bands(fine_lon, fine_lat, surface, levels) if fine_lon is not None else []

    comparison_df = None
    comparison_bands = None
    comparison_label = None
    if args.weekend_compare:
        comparison_dt = compute_comparison_datetime(departure_dt)
        comparison_label = "weekend" if comparison_dt.weekday() >= 5 else "weekday"
        logger.info("Running comparison query for %s (%s)", comparison_dt.isoformat(), comparison_label)
        try:
            comparison_df = run_travel_time_queries(
                gmaps, cache, rate_limiter, grid_points, destination,
                int(comparison_dt.timestamp()), args.reverse, transit_modes,
                args.max_workers, args.batch_size, args.no_resume,
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted during comparison query; primary outputs are still written.")
            comparison_df = None
        if comparison_df is not None:
            write_csv(comparison_df, output_dir / "travel_times_comparison.csv")
            write_geojson(build_geojson(comparison_df), output_dir / "travel_times_comparison.geojson")
            _, _, comparison_surface = interpolate_surface(comparison_df, bounds)
            if comparison_surface is not None:
                comparison_bands = compute_isochrone_bands(fine_lon, fine_lat, comparison_surface, levels)

    departure_label = departure_dt.strftime("%A %Y-%m-%d %H:%M %Z")
    m = build_map(
        df, bands, destination, departure_label, args,
        comparison_df=comparison_df, comparison_bands=comparison_bands, comparison_label=comparison_label,
    )
    map_path = output_dir / "transit_map.html"
    m.save(str(map_path))
    logger.info("Wrote %s", map_path)

    cache.close()
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:  # noqa: BLE001
        logger.exception("Fatal error")
        sys.exit(1)
