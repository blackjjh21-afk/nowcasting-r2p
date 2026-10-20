"""Gauge-only ordinary kriging of RN60, following Shehu (2021), Sect. 3.1.1.

The seasonal exponential semivariogram is fitted locally, never copied from
the German case study. Each wet, spatially variable timestep's experimental
semivariogram is divided by that timestep's station variance before averaging
over the season. No logarithmic transform or radar covariate is used.

Implementation extensions are explicit: finite nonnegative observations only;
co-located gauges averaged; conservative geographic support; and negative OK
estimates clamped to zero by default. The returned kriging variance describes
the *unclipped linear estimator*, not a calibrated rainfall confidence interval.
Coordinates and distances are in a local AEQD projection, in kilometres.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from scipy.optimize import least_squares
from scipy.spatial import Delaunay, QhullError
from scipy.spatial.distance import cdist, pdist



@dataclass(frozen=True)
class ExponentialVariogram:
    nugget: float
    partial_sill: float
    scale_km: float

    def __post_init__(self):
        parameters = np.asarray([self.nugget, self.partial_sill, self.scale_km])
        if not np.isfinite(parameters).all() or self.nugget < 0 or self.partial_sill <= 0 or self.scale_km <= 0:
            raise ValueError("variogram needs finite nugget >= 0, partial_sill > 0, scale_km > 0")

    def semivariance(self, distance_km):
        distance = np.asarray(distance_km, dtype=np.float64)
        if np.any(distance < 0):
            raise ValueError("distance must be nonnegative")
        # The nugget is a discontinuity at the origin, not gamma(0).
        return np.where(distance > 0, self.nugget + self.partial_sill * (-np.expm1(-distance / self.scale_km)), 0.0)

    def covariance(self, distance_km):
        distance = np.asarray(distance_km, dtype=np.float64)
        return self.partial_sill * np.exp(-distance / self.scale_km) + self.nugget * (distance <= 1e-9)

    def as_dict(self):
        return {"model": "exponential", "nugget": float(self.nugget),
                "partial_sill": float(self.partial_sill), "scale_km": float(self.scale_km),
                "effective_range_km": float(3 * self.scale_km),
                "range_convention": "gamma(h)=c0+c*(1-exp(-h/a)); scale_km=a; effective_range_km=3*a (95.02% of partial sill)",
                "normalized_by_timestep_variance": True}


def _valid_values(values):
    values = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(values) & (values >= 0), values, np.nan)


def _unique_sites(xy_km):
    xy = np.asarray(xy_km, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("station_xy_km must be finite [station, 2]")
    # Millimetre coordinate tolerance prevents numerically duplicated sites.
    _, first, inverse = np.unique(np.round(xy, 6), axis=0, return_index=True, return_inverse=True)
    return xy[first], inverse


def _pool_sites(values, inverse, n_sites):
    values = _valid_values(values)
    finite = np.isfinite(values)
    counts = np.bincount(inverse[finite], minlength=n_sites)
    sums = np.bincount(inverse[finite], weights=values[finite], minlength=n_sites)
    return np.divide(sums, counts, out=np.full(n_sites, np.nan), where=counts > 0)


def fit_variogram_from_arrays(
    station_xy_km, observations, *, nlags=20, cutoff_km=100.0,
    variance_ddof=1, min_gauges=3,
):
    """Fit seasonal normalized empirical gamma from [time, station] samples.

    Every timestep contributes equal weight to each nonempty lag bin. The
    final fit uses unweighted bounded nonlinear least squares across bins.
    Caller is responsible for selecting the permitted years and stations.
    """
    if nlags < 3 or cutoff_km <= 0 or variance_ddof not in (0, 1) or min_gauges < 3:
        raise ValueError("invalid lag, cutoff, variance_ddof, or min_gauges setting")
    xy, inverse = _unique_sites(station_xy_km)
    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != len(inverse):
        raise ValueError("observations must be [time, station]")
    i, j = np.triu_indices(len(xy), 1)
    distances = pdist(xy)
    pair_keep = (distances > 0) & (distances <= cutoff_km)
    i, j, distances = i[pair_keep], j[pair_keep], distances[pair_keep]
    if len(distances) == 0:
        raise ValueError("no distinct station pairs within the lag cutoff")
    edges = np.linspace(0, cutoff_km, nlags + 1)
    bins = np.minimum(np.searchsorted(edges, distances, side="right") - 1, nlags - 1)
    gamma_sum, time_count = np.zeros(nlags), np.zeros(nlags, dtype=np.int64)
    lag_sum, pair_count = np.zeros(nlags), np.zeros(nlags, dtype=np.int64)
    selected_count = 0
    for source_row in observations:
        row = _pool_sites(source_row, inverse, len(xy))
        finite = np.isfinite(row)
        if finite.sum() < min_gauges:
            continue
        variance = np.var(row[finite], ddof=variance_ddof)
        if variance <= 1e-12 or np.max(row[finite]) <= 0:
            continue
        good = finite[i] & finite[j]
        count = np.bincount(bins[good], minlength=nlags)
        if not np.any(count):
            continue
        gamma = (row[i[good]] - row[j[good]]) ** 2 / (2 * variance)
        total = np.bincount(bins[good], weights=gamma, minlength=nlags)
        occupied = count > 0
        gamma_sum[occupied] += total[occupied] / count[occupied]
        time_count[occupied] += 1
        lag_sum += np.bincount(bins[good], weights=distances[good], minlength=nlags)
        pair_count += count
        selected_count += 1
    occupied = time_count > 0
    if occupied.sum() < 3:
        raise ValueError("fewer than three populated lag bins; cannot fit an exponential variogram")
    lag = lag_sum[occupied] / pair_count[occupied]
    empirical = gamma_sum[occupied] / time_count[occupied]
    initial = [max(0.0, min(float(empirical.min()) / 2, 0.5)),
               max(float(empirical.max()), 0.1), cutoff_km / 6]
    fitted = least_squares(
        lambda p: p[0] + p[1] * (-np.expm1(-lag / p[2])) - empirical,
        initial, bounds=([0, 1e-8, 0.05], [10, 10, 3 * cutoff_km]),
        xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=10000,
    )
    if not fitted.success or not np.isfinite(fitted.x).all():
        raise RuntimeError(f"variogram fit failed: {fitted.message}")
    model = ExponentialVariogram(*fitted.x)
    return {"variogram": model.as_dict(), "fit": {
        "variance_ddof": variance_ddof, "nlags": nlags, "cutoff_km": cutoff_km,
        "min_gauges": min_gauges, "n_wet_variable_timesteps": selected_count,
        "n_unique_sites": len(xy), "lag_edges_km": edges.tolist(),
        "lag_centres_km": lag.tolist(), "empirical_normalized_gamma": empirical.tolist(),
        "timesteps_per_bin": time_count[occupied].tolist(), "pairs_per_bin": pair_count[occupied].tolist(),
        "least_squares_rmse": float(np.sqrt(np.mean(fitted.fun ** 2))),
        "parameters_at_bound": bool(np.any(fitted.active_mask)),
        "effective_range_exceeds_lag_cutoff": bool(3 * model.scale_km > cutoff_km),
        "range_caution": "An effective range beyond the fitted lag cutoff is extrapolated and may be weakly identified; it is not a measured decorrelation distance.",
        "averaging": "equal weight per timestep per populated lag bin",
        "fit_weighting": "unweighted nonlinear least squares across populated bins",
    }}


class OrdinaryKriging:
    """Reusable vectorized solver; weights cached by finite station subset."""

    def __init__(self, station_xy_km, target_xy_km, variogram, *,
                 min_neighbors=3, support_radius_km=30.0, max_nearest_km=20.0,
                 require_hull=True, variance_ddof=1, negative_policy="clip_zero",
                 max_cache_entries=4, chunk_size=1024, condition_limit=1e12):
        self.xy, self.inverse = _unique_sites(station_xy_km)
        self.target = np.asarray(target_xy_km, dtype=np.float64)
        if self.target.ndim != 2 or self.target.shape[1] != 2 or not np.isfinite(self.target).all():
            raise ValueError("target_xy_km must be finite [target, 2]")
        if min_neighbors < 3 or support_radius_km <= 0 or max_nearest_km <= 0:
            raise ValueError("support requires at least three gauges and positive distances")
        if variance_ddof not in (0, 1) or negative_policy not in ("clip_zero", "retain"):
            raise ValueError("invalid variance_ddof or negative_policy")
        if max_cache_entries < 1 or chunk_size < 1 or condition_limit <= 1:
            raise ValueError("invalid numerical solver configuration")
        self.model = variogram
        self.min_neighbors, self.radius, self.nearest = min_neighbors, support_radius_km, max_nearest_km
        self.require_hull, self.ddof, self.negative_policy = require_hull, variance_ddof, negative_policy
        self.max_cache_entries, self.chunk_size, self.condition_limit = max_cache_entries, chunk_size, condition_limit
        self._cache = OrderedDict()

    def _weights(self, finite):
        key = np.packbits(finite).tobytes()
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        xy = self.xy[finite]
        n = len(xy)
        distance = cdist(self.target, xy)
        supported = (np.sum(distance <= self.radius, axis=1) >= self.min_neighbors) & (distance.min(axis=1) <= self.nearest)
        if self.require_hull:
            try:
                supported &= Delaunay(xy).find_simplex(self.target, tol=1e-10) >= 0
            except QhullError:
                supported[:] = False
        covariance = self.model.covariance(cdist(xy, xy))
        eigenvalues = eigvalsh(covariance, check_finite=True)
        sill = self.model.nugget + self.model.partial_sill
        jitter, condition = 0.0, np.inf
        for relative in (0.0, 1e-12, 1e-10, 1e-8, 1e-6):
            jitter = relative * sill
            minimum, maximum = eigenvalues[0] + jitter, eigenvalues[-1] + jitter
            condition = float(maximum / minimum) if minimum > 0 else np.inf
            if minimum > 0 and condition <= self.condition_limit:
                break
        else:
            raise np.linalg.LinAlgError(f"kriging covariance is ill-conditioned ({condition:g})")
        factor = cho_factor(covariance + jitter * np.eye(n), lower=True, check_finite=True)
        inverse_one = cho_solve(factor, np.ones(n))
        denominator = float(inverse_one.sum())
        if not np.isfinite(denominator) or denominator <= 0:
            raise np.linalg.LinAlgError("invalid ordinary-kriging constraint denominator")
        locations = np.flatnonzero(supported)
        weights = np.empty((len(locations), n), dtype=np.float64)
        variance = np.empty(len(locations), dtype=np.float64)
        for start in range(0, len(locations), self.chunk_size):
            stop = start + self.chunk_size
            cross = self.model.covariance(distance[locations[start:stop]]).T
            inverse_cross = cho_solve(factor, cross)
            correction = (1 - inverse_cross.sum(axis=0)) / denominator
            w = inverse_cross + inverse_one[:, None] * correction
            weights[start:stop] = w.T
            # Use the ORIGINAL covariance, not the jitter-inflated covariance.
            variance[start:stop] = sill - np.sum(w * cross, axis=0) + correction - jitter * np.sum(w * w, axis=0)
        if not np.isfinite(weights).all() or not np.isfinite(variance).all():
            raise np.linalg.LinAlgError("nonfinite kriging weights or variance")
        if not np.allclose(weights.sum(axis=1), 1, atol=1e-9, rtol=0):
            raise np.linalg.LinAlgError("ordinary-kriging weights do not sum to one")
        if np.any(variance < -1e-7 * sill):
            raise np.linalg.LinAlgError("negative kriging variance beyond floating-point tolerance")
        result = (weights, np.maximum(variance, 0), supported, condition, jitter)
        self._cache[key] = result
        if len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        return result

    def predict(self, observations):
        observations = np.asarray(observations)
        if observations.shape != (len(self.inverse),):
            raise ValueError("one observation is required per input station")
        row = _pool_sites(observations, self.inverse, len(self.xy))
        finite = np.isfinite(row)
        count = int(finite.sum())
        prediction, variance = np.full(len(self.target), np.nan), np.full(len(self.target), np.nan)
        result = {"rainfall_mm": prediction, "kriging_variance_mm2": variance,
                  "valid_mask": np.zeros(len(self.target), dtype=bool), "n_gauges": count,
                  "n_input_gauges_finite": int(np.isfinite(_valid_values(observations)).sum()),
                  "n_negative_clipped": 0, "status": "insufficient_gauges"}
        if count < self.min_neighbors:
            return result
        weights, normalized_variance, supported, condition, jitter = self._weights(finite)
        station_variance = float(np.var(row[finite], ddof=self.ddof))
        estimate = weights @ row[finite]
        if self.negative_policy == "clip_zero":
            result["n_negative_clipped"] = int(np.sum(estimate < 0))
            estimate = np.maximum(estimate, 0)
        prediction[supported], variance[supported] = estimate, normalized_variance * station_variance
        result.update(valid_mask=supported.copy(), station_variance_mm2=station_variance,
                      condition_number=condition, jitter=jitter,
                      status="ok" if supported.any() else "no_spatial_support")
        return result

    def close(self):
        self._cache.clear()


