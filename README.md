# Direct R2P

Code for *Gauge-referenced hourly rainfall nowcasting in South Korea:
Comparing direct radar-to-point and field-first forecast routes*,
by Jaehee Jung and Joon-Woo Roh.

Direct R2P predicts hourly rainfall at target locations directly from radar
and gauge histories. We compare it with pySTEPS, adapted exPreCast and
NowcastNet, each followed by a gauge-supervised CNN readout.

## Installation

```bash
conda env create -f environment.yml
conda activate r2p-4km10min-release
python -m pip install --no-deps -e .
```

The environment uses Python 3.12 and PyTorch 2.6 with CUDA 12.4.

## Data

Radar and gauge observations are available from the
[KMA API Hub](https://apihub.kma.go.kr/). The models use radar fields at
4-km resolution and 10-min intervals, gauge rainfall histories and station
coordinates. See [input formats](docs/INPUTS.md) for file layouts, preprocessing
and evaluation inputs.

## Training and inference

### Direct R2P

Set the data paths, then train and predict. This Bash example uses seed 0:

```bash
inputs=(
  --project-root .
  --radar-root data/private/radar_4km10min_tiff
  --gauge-csv data/private/gauge_rn60_rn15.csv
  --mapping-csv data/private/station_contract/mapping.csv
  --stations-csv data/private/station_contract/stations.csv
  --split-csv data/private/station_contract/split.csv
  --output-root outputs/direct_r2p
  --device cuda:0
)
r2p-4km10min train "${inputs[@]}" --seeds 0
r2p-4km10min evaluate "${inputs[@]}" --seeds 0 \
  --checkpoint-name epoch_027.pt --evaluation-tag csi3
```

The paper uses epochs 27, 14 and 21 for seeds 0, 1 and 2, respectively.
`r2p-select-csi3` selects epochs from saved validation scores using mean CSI
at 1, 5 and 10 mm. Its command options are listed by `r2p-select-csi3 --help`.

### Field-first routes

Generate radar-field forecasts with pySTEPS, adapted exPreCast or NowcastNet,
then fit a CNN readout to the station patches. The commands below list the
options for each step:

| Step | Command |
|---|---|
| pySTEPS forecasts and patches | `pysteps-4km10min --help` |
| exPreCast training and field export | `exprecast-adapted --help` |
| exPreCast checkpoint selection | `exprecast-select-rn60 --help` |
| exPreCast CNN inputs | `exprecast-export-adapter --help` |
| NowcastNet training | `nowcastnet-hsr-field --help` |
| NowcastNet field export | `nowcastnet-export-bank --help` |
| NowcastNet CNN inputs | `nowcastnet-cnn-prepare --help` |
| CNN epoch selection, fitting and prediction | `cnn-readout-4km10min --help` |

NowcastNet settings are in [configs/nowcastnet_hsr_field.json](configs/nowcastnet_hsr_field.json).
The exPreCast adapter uses the [official model source](https://github.com/tony890048/exPreCast)
at revision `092922c126bdf6098fbda2e08b2f1f3b2873bd9a`.

## Evaluation

Prepare the observation and prediction arrays described in
[input formats](docs/INPUTS.md), then run:

```bash
jhrs-verify manifest.json --output verification --bootstrap 2000
```

This calculates station and catchment scores, paired confidence intervals
and heavy-rain peak errors. `r2p-radar-only` and `r2p-station-dropout` evaluate
the gauge-input and station-dropout comparisons; use `--help` for their inputs.

Run the tests with:

```bash
pytest -q tests
```

## Citation and license

Software archive: [10.5281/zenodo.23489108](https://doi.org/10.5281/zenodo.23489108).
Citation details are in [CITATION.cff](CITATION.cff).

Original code is licensed under [BSD-3-Clause](LICENSE).
Third-party code retains its respective licenses; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
