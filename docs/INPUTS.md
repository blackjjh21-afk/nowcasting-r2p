# Input formats

Radar and gauge observations come from the [KMA API Hub](https://apihub.kma.go.kr/).
Paths below are relative to the repository root.

## Times and units

Rainfall accumulations are in mm and rain rates in mm/h. Array timestamps are
KST-naive `int64` nanoseconds representing local wall-clock time, for example
`np.datetime64("2024-06-01T00:00", "ns").astype("int64")`.
Issue axes are increasing and unique on the 10-minute grid. Use the same
issue, station and lead ordering when combining arrays.

## Radar

The HSR preparation command reads files named with `RDR_CMP_HSR` and a
`YYYYMMDDHHMM` timestamp, ending in `.bin` or `.bin.gz`. Raw fields are
`2881 × 2305` little-endian `int16` values in `100 × dBZ`; the coordinate
NetCDF contains `lon` and `lat` arrays of the same shape.

```bash
prepare-kma-hsr-4km10min \
  --raw-root data/private/hsr_raw \
  --coordinate-netcdf data/private/rdr_500m_latlon.nc \
  --output-root data/private/radar_4km10min_tiff \
  --years 2019-2025 --workers 4
```

The resulting directory contains `YYYY/MM/DD/YYYYMMDDHHMM.tiff`,
`grid_coordinates.npz`, `frames.csv` and `manifest.json`. Keep these files
together: the loaders use the grid coordinates, source-support mask and
preprocessing metadata. TIFFs are `float32 [256,256]`, with 4-km spacing and
values `max(raw_100xdbz, 0) / 10000`. Seven frames cover −60 to 0 min.

## Gauge and station files

```text
data/private/
├── radar_4km10min_tiff/
├── gauge_rn60_rn15.csv
└── station_contract/
    ├── mapping.csv
    ├── stations.csv
    └── split.csv
```

| File | Columns |
|---|---|
| `mapping.csv` | `station_id,exprecast_y,exprecast_x,exprecast_nearest_distance_km` |
| `stations.csv` | `station_id,name,target_col,rn15m_col,iy,ix,lat,lon,split` |
| `split.csv` | `station_id,split` |
| `gauge_rn60_rn15.csv` | `Year,Month,Day,Hour,Minute`, followed by each station's RN60 and RN15 columns |

The three station files share 642 unique integer IDs: 514 `train` and 128
`test`. `target_col` and `rn15m_col` name columns in the gauge CSV; gauge rows
are increasing 5-minute KST observations, with empty fields or `NaN` for missing values.
`exprecast_y,x` are zero-based indices on the prepared radar grid.
`iy,ix` refer to the original 2-km station geometry; attention uses
`(x,y) = (2*ix,2*iy)` km. Coordinates `lat,lon` are in degrees, and mapping
distances are in km.

Direct R2P checks the paper's station-file hashes by default. Use
`--allow-unfrozen-contract` for newly prepared CSVs; they still need the
642-station split, source-backed grid cells and mapping distances ≤2.761 km.
Keep the training output directory intact when selecting checkpoints or
resuming a run, including `scientific_contract.json` and saved scaling data.
Use trusted checkpoints for Direct R2P, exPreCast and NowcastNet: their
training files contain Python state as well as tensors.

## Forecast fields

Field forecasts have 18 steps at +10,…,+180 min.

| Source | Input/output format |
|---|---|
| pySTEPS | `forecast` takes a normalized HSR array `[7,H,W]` (`.npy`, or an NPZ key selected with `--input-key`) and `--issue-time-ns`. Its NPZ carries `forecast_normalized_hsr`, `forecast_rain_rate_mm_h`, `lead_minutes` and `issue_times_ns`. |
| exPreCast | Exported HDF5 contains `forecast_normalized_dbz [N,18,256,256]`, `issue_time_ns [N]` and `lead_minutes [18]`. Keep the generated `completed` array and `contract_json` attribute with the export. |
| NowcastNet | The exported bank contains `mean_rate_mmph.npy` (`float32 [N,18,256,256]`), `issue_times_kst_ns.npy`, `spatial_mask.npy`, `contract.json` and `status.json`. |

The exPreCast loader uses upstream revision
`092922c126bdf6098fbda2e08b2f1f3b2873bd9a`; pass that checkout with
`--official-repo`. NowcastNet paths are configured in
[`configs/nowcastnet_hsr_field.json`](../configs/nowcastnet_hsr_field.json).
For an explicitly selected NowcastNet checkpoint, supply `--checkpoint` and
`--checkpoint-sha256`, keeping its original `contract.json` beside the checkpoint.

## CNN arrays

The CNN workflow reads an NPZ with scalar `schema="cnn_readout_prepared_npz_v1"`.
Here `N` is issuance time, `S` station, and `L=13` for leads +60,…,+180 min.

| Key | Shape / contents |
|---|---|
| `patches` | `float32 [N,S,L,6,3,3]`: normalized-reflectivity patches at `L−50,…,L` |
| `issue_times_ns` | `int64 [N]` |
| `station_ids`, `station_folds` | Integer `[S]`; fixed folds 0, 1 or 2 |
| `lead_minutes` | Integer `[13]`: `60,70,…,180` |
| `targets_mm` | `float32 [N,S,L]` for fitting/selection; nonnegative RN60 or `NaN` |
| `auxiliary` | `float32 [N,S,L,39]`: normalized x, y, lead/180 and 36 gauge-context features |
| `oof_auxiliary` | `float32 [3,N,S,L,39]` for OOF selection; entry `f` uses context excluding station fold `f` |

Patches and context features must be finite. Gauge context uses fitting-station
histories available at issuance. Supply `auxiliary` for gauge-context readouts;
its omission selects an all-zero context. Prediction inputs can omit targets.

`pysteps-4km10min patches` produces the patch and axis arrays.
`exprecast-export-adapter extract-patches` writes `patches18_f16.npy` and its
time/station axes; `exprecast_adapter.adapter.gather_scene_windows` converts the
18-step patches to `[N,S,13,6,3,3]`. Add the schema, station folds, contexts
and training targets when assembling the CNN NPZ.

`nowcastnet-cnn-prepare` combines a field bank with a context NPZ:

| Context key | Shape / value |
|---|---|
| `schema` | Scalar `"nowcastnet_cnn_context_v1"` |
| `issue_times_ns` | Same `int64 [N]` axis as the bank |
| `station_ids`, `station_folds` | Integer `[S]` |
| `grid_y`, `grid_x` | Integer `[S]` patch centers, 1–254 |
| `xy_norm`, `context36` | Finite `[S,2]` and `[N,S,36]` |
| `context_fitting_stations_only`, `context_available_at_issuance` | Scalar `True` |
| `targets_mm` | `[N,S,13]` for fitting/selection |
| `oof_context36`, `oof_context_excludes_held_fold` | `[3,N,S,36]` and scalar `True` for selection |

## Verification

`jhrs-verify` reads a JSON manifest and numeric NPY arrays stored alongside it.
This two-site example shows the layout; set `site_ids` and `expected_site_count`
to match your station or catchment arrays.

```json
{
  "schema": "jhrs-verification-v1",
  "kind": "catchment",
  "timezone": "Asia/Seoul",
  "timestamp_encoding": "naive_kst_nanoseconds",
  "issues": "issues.npy",
  "valid_times": "valid_times.npy",
  "observations": "observations.npy",
  "lead_minutes": [60, 90, 120, 150, 180],
  "site_ids": ["catchment_01", "catchment_02"],
  "expected_site_count": 2,
  "years": [2024, 2025],
  "months": [6, 7, 8, 9],
  "routes": [
    {"name": "Direct R2P", "member_ids": [0, 1, 2], "predictions": "direct.npy"},
    {"name": "pySTEPS + CNN", "member_ids": [0, 1, 2], "predictions": "pysteps.npy"},
    {"name": "exPreCast + CNN", "member_ids": [0, 1, 2], "predictions": "exprecast.npy"},
    {"name": "NowcastNet + CNN", "member_ids": [0, 1, 2], "predictions": "nowcastnet.npy"}
  ]
}
```

Observations are floating `[valid_time,site]`; each route's predictions are
floating `[issue,member,lead,site]`. Store three fitted members separately.
For CNN outputs `[N,S,L]`, stack the three `predictions_mm` arrays on axis 1
and transpose to `[N,3,L,S]`. Select the five listed leads from each model's
`lead_minutes` array before verification. Direct R2P stores each seed's
predictions in `preds_mm.npy`, alongside its time, station and lead axes.
The reference time axis must include every issuance time plus each lead.
Use `"kind":"station"` for station scores. An optional `observation_valid`
file contains a Boolean mask with the observations' shape.

For cell-level catchment inputs, add `"area_weights":"area_weights.npy"`.
Replace the site axis with cells and supply `[catchment,cell]` weights equal
to cell–catchment overlap area divided by catchment area, with row sums of one.
The `catchment_kriging` routines take projected coordinates in km and RN60
gauge arrays to produce cell references; catchment boundaries come from
[WAMIS](https://map.wamis.go.kr/).
