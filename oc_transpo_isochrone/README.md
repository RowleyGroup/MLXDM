# OC Transpo Transit Isochrone Map

A self-contained Python application that generates a colour-coded, interactive
real-estate-style commute-time map for public transit (OC Transpo / STO) across
Ottawa-Gatineau, using the Google Maps Platform APIs.

Given a destination address and a departure date/time, it samples a grid of points
across Ottawa, Gatineau, Kanata, Orleans, Barrhaven, Stittsville, Nepean, and Aylmer,
queries the transit travel time from each point to the destination, and renders an
interactive Leaflet/Folium HTML map with coloured points and smooth isochrone-band
polygons (0-15, 15-30, 30-45, 45-60, 60-75, 75-90, and >90 minutes).

## Installation

Requires Python 3.12 on Ubuntu (also works on any recent Python 3.10+).

```bash
cd oc_transpo_isochrone
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Google API setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create (or
   select) a project.
2. Enable **billing** on the project (both APIs below require it; Google provides a
   recurring monthly free credit that comfortably covers casual/one-off use).
3. Under **APIs & Services > Library**, enable:
   - **Geocoding API** (turns the destination address into coordinates)
   - **Distance Matrix API** (computes transit travel times; this is the current,
     supported Google Maps Platform routing API for many-origins-to-one-destination
     queries and is not deprecated)
4. Under **APIs & Services > Credentials**, create an **API key**. It is strongly
   recommended to restrict it (API restrictions: Geocoding API + Distance Matrix API;
   optionally IP/application restrictions for the machine running the script).
5. Export the key so the script can find it:

   ```bash
   export GOOGLE_MAPS_API_KEY="your-api-key-here"
   ```

   (or pass `--api-key your-api-key-here` on the command line instead).

Note on transit routing: Google's Distance Matrix API requires `departure_time` to be
now or in the future for `mode=transit` — you cannot query historical transit times.

## Usage

```bash
python transit_isochrone_map.py \
    --destination "1125 Colonel By Dr Ottawa ON" \
    --departure "2026-09-01 08:00" \
    --grid-km 1.0 \
    --max-minutes 90
```

This writes, by default, into `./output/`:

- `travel_times.csv` — raw grid samples: `latitude, longitude, travel_time_minutes, status`
- `travel_times.geojson` — the same samples as a GeoJSON `FeatureCollection` of points
- `transit_map.html` — the interactive map (open directly in a browser)
- `run.log` — a log of the run
- `transit_cache.sqlite3` — the local cache (safe to delete to force a full re-query)

### Resuming an interrupted run

Every completed batch of API queries is written to the SQLite cache immediately. If the
script is interrupted (Ctrl-C, network outage, crash), simply re-run the **exact same
command** — grid points that were already successfully queried are read from the cache
instead of being re-queried, and only the remaining points cost API quota.

### All options

| Flag | Default | Description |
|---|---|---|
| `--destination` | *(required)* | Destination street address |
| `--departure` | *(required)* | Local (America/Toronto) departure time, `YYYY-MM-DD HH:MM` |
| `--grid-km` | `1.0` | Sample grid spacing in kilometres |
| `--max-minutes` | `90` | Travel-time cap used for the colour scale/heatmap |
| `--output-dir` | `./output` | Output directory |
| `--api-key` | *(env `GOOGLE_MAPS_API_KEY`)* | Google Maps API key |
| `--cache-db` | `<output-dir>/transit_cache.sqlite3` | SQLite cache path |
| `--max-workers` | `4` | Concurrent API request threads |
| `--requests-per-second` | `8.0` | Max Distance Matrix requests/second (rate limit) |
| `--batch-size` | `25` | Grid points per Distance Matrix request (Google's max is 25) |
| `--transit-modes` | *(any)* | Comma list from `bus,subway,train,tram,rail` |
| `--reverse` | off | Compute travel time **from** the destination **to** each grid point instead of to it |
| `--heatmap` | off | Add a toggleable proximity heatmap layer |
| `--color-scheme` | `bands` | `bands` (fixed 15-min bands) or `continuous` (gradient) |
| `--weekend-compare` | off | Also compute a second layer for the same time on the nearest day of the opposite weekday/weekend category |
| `--region-bounds` | *(NCR default)* | Override sample bounding box: `min_lat,min_lon,max_lat,max_lon` |
| `--no-resume` | off | Ignore the cache and re-query everything |
| `--log-level` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |

### Example: reverse mode + heatmap + weekend comparison

```bash
python transit_isochrone_map.py \
    --destination "40 Elgin St, Ottawa, ON" \
    --departure "2026-09-08 08:30" \
    --grid-km 0.75 \
    --max-minutes 60 \
    --reverse \
    --heatmap \
    --weekend-compare \
    --color-scheme continuous
```

## How it works

1. **Geocoding** — the destination address is resolved once via the Geocoding API and
   cached.
2. **Grid** — a regular lat/lon grid at `--grid-km` spacing is generated over a
   bounding box that covers Ottawa, Gatineau, Kanata, Orleans, Barrhaven, Stittsville,
   Nepean, and Aylmer (validated at start-up against reference points for each region).
3. **Routing** — grid points are batched (up to 25 per request, Google's limit for the
   Distance Matrix API) and queried concurrently across a thread pool, paced by a
   token-based rate limiter. Every response is cached in SQLite immediately.
4. **Isochrones** — the sparse point samples are interpolated onto a fine mesh (linear
   interpolation inside the sample area, nearest-neighbour outside it) and Matplotlib's
   contouring engine is used purely as a geometry tool to extract filled polygons for
   each travel-time band, which are cleaned up and simplified with Shapely.
5. **Map** — Folium renders an interactive Leaflet map: coloured sample points with
   hover tooltips, semi-transparent isochrone-band polygons, a destination marker, a
   fixed legend, an optional heatmap layer, and (if requested) a toggleable
   weekday/weekend comparison layer.

## Notes / limitations

- Grid resolution and area trade off directly against Google API cost and run time:
  the default 1 km grid over the full NCR is ~2,000 points (~80 batched requests).
  Halving `--grid-km` roughly quadruples both.
- The isochrone polygons are a smoothed *interpolation* of sparse point samples, not an
  exact routed boundary — this matches how real-estate commute-time tools work, but
  fine local variation (e.g. one street with an unusually fast bus route) will be
  smoothed over.
- `tests/` in the parent MLXDM repository is unrelated to this tool (it exercises the
  neural-network-potential package in this repository); this application has no
  dependency on the rest of the repository.
