"""Distance-dependent fluorescence-quenching analysis.

This module operates on distance time series that have already been calculated
from molecular-dynamics trajectories.  Its first supported input is the
``MinimumContactDistanceResult`` produced by :mod:`mdsim.analysis`, but the
implementation uses duck typing so that compatible result objects or mappings
can also be processed without importing the analysis module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, TypeAlias, Union

import numpy as np


@dataclass(frozen=True)
class QuenchingDecayResult:
    """Distance-dependent quenching hazard for a set of chain time series.

    ``decay_exponent_per_chain`` has shape ``(n_chains, n_frames)``.  Each
    element is the dimensionless interval hazard ``q(r) * dt`` calculated from
    the corresponding minimum-contact distance.
    """

    time_ps: np.ndarray  # (n_frames,), physical time after kinetic scaling
    decay_exponent_per_chain: np.ndarray  # (n_chains, n_frames)
    chain_labels: tuple[str, ...]
    n_chains: int
    n_frames: int
    delta_t_fast_ps: float
    delta_t_physical_ps: float
    q0_per_s: float
    beta: float
    a0: float
    eps: Optional[float]
    kinetic_scale: float
    distance_field: str = "distance_per_chain_nm"
    source_frame_start: int = 0
    source_frame_stop: Optional[int] = None

    @property
    def exponent_per_chain(self) -> np.ndarray:
        """Alias for :attr:`decay_exponent_per_chain`."""
        return self.decay_exponent_per_chain

    @property
    def hazard_per_chain(self) -> np.ndarray:
        """Alias emphasizing that the values are interval hazards."""
        return self.decay_exponent_per_chain

    @property
    def series(self) -> tuple[np.ndarray, ...]:
        """Return one ``(time_ps, exponent)`` array per chain.

        This provides the same per-trajectory array layout used by the older
        file-based quenching workflow while retaining chain labels and a compact
        two-dimensional representation in the main result object.
        """
        return tuple(np.column_stack((self.time_ps, row)) for row in self.decay_exponent_per_chain)


@dataclass(frozen=True)
class QuenchingCurveResult:
    """Time-origin-averaged survival curves for each input chain.

    ``survival_per_chain`` and ``counts_per_chain`` have shape
    ``(n_chains, n_lag_bins)``.  The values reproduce :func:`quench` applied
    independently to every chain in a :class:`QuenchingDecayResult`.
    """

    lag_time_ps: np.ndarray  # (n_lag_bins,)
    survival_per_chain: np.ndarray  # (n_chains, n_lag_bins)
    counts_per_chain: np.ndarray  # (n_chains, n_lag_bins)
    chain_labels: tuple[str, ...]
    n_chains: int
    n_frames: int
    n_lag_bins: int
    tres_ps: float

    @property
    def time_ps(self) -> np.ndarray:
        """Alias for :attr:`lag_time_ps`."""
        return self.lag_time_ps

    @property
    def quenching_curve_per_chain(self) -> np.ndarray:
        """Alias for :attr:`survival_per_chain`."""
        return self.survival_per_chain

    @property
    def series(self) -> tuple[np.ndarray, ...]:
        """Return one ``(lag time, mean survival, count)`` array per chain."""
        return tuple(
            np.column_stack((self.lag_time_ps, values, counts))
            for values, counts in zip(self.survival_per_chain, self.counts_per_chain)
        )


QuenchingInput: TypeAlias = Union[Any, Mapping[Any, "QuenchingInput"]]
QuenchingOutput: TypeAlias = Union[
    QuenchingDecayResult,
    QuenchingCurveResult,
    dict[Any, "QuenchingOutput"],
]


def _finite_scalar(value: Any, *, name: str) -> float:
    """Convert one scalar parameter to a finite float."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _result_has_field(result: Any, field: str) -> bool:
    if isinstance(result, Mapping):
        return field in result
    return hasattr(result, field)


def _result_field(result: Any, field: str) -> Any:
    if isinstance(result, Mapping):
        if field not in result:
            raise KeyError(f"result does not contain field {field!r}")
        return result[field]
    if not hasattr(result, field):
        raise AttributeError(f"result has no attribute {field!r}")
    return getattr(result, field)


def _chain_labels(result: Any, n_chains: int) -> tuple[str, ...]:
    if _result_has_field(result, "chain_labels"):
        labels = tuple(str(label) for label in _result_field(result, "chain_labels"))
        if len(labels) != n_chains:
            raise ValueError(f"chain_labels contains {len(labels)} labels, expected {n_chains}")
        if len(set(labels)) != len(labels):
            raise ValueError("chain_labels contains duplicate labels")
        return labels
    return tuple(f"chain {index}" for index in range(n_chains))


def _apply_decay_to_result(
    result: Any,
    *,
    q0: float,
    beta: float,
    a0: float,
    delta_t_fast_ps: float,
    eps: Optional[float],
    kinetic_scale: float,
    distance_field: str,
    time_start_fast_ps: float,
    time_stop_fast_ps: Optional[float],
) -> QuenchingDecayResult:
    distances_full = np.asarray(_result_field(result, distance_field), dtype=np.float64)
    if distances_full.ndim != 2:
        raise ValueError(
            f"{distance_field!r} must have shape (n_chains, n_frames); "
            f"got {distances_full.shape}"
        )

    n_chains, n_frames_full = distances_full.shape
    if n_chains < 1:
        raise ValueError(f"{distance_field!r} contains no chains")
    if n_frames_full < 2:
        raise ValueError(
            f"{distance_field!r} must contain at least two frames to define a timestep"
        )

    # Match the time-window convention used by condensate_analysis: include
    # frames at t >= time_start and t < time_stop.  A small tolerance avoids
    # skipping exact multiples because of floating-point roundoff.
    tol = 1.0e-12
    frame_start = int(math.ceil(time_start_fast_ps / delta_t_fast_ps - tol))
    frame_start = max(0, min(frame_start, n_frames_full))
    if time_stop_fast_ps is None:
        frame_stop = n_frames_full
    else:
        frame_stop = int(math.ceil(time_stop_fast_ps / delta_t_fast_ps - tol))
        frame_stop = max(0, min(frame_stop, n_frames_full))

    if frame_stop <= frame_start:
        raise ValueError(
            "the requested time range selects fewer than one frame "
            f"(resolved slice {frame_start}:{frame_stop} for {n_frames_full} frames)"
        )

    distances = distances_full[:, frame_start:frame_stop]
    n_frames = int(distances.shape[1])
    if n_frames < 2:
        raise ValueError("the requested time range must contain at least two frames for quenching")

    labels = _chain_labels(result, n_chains)

    # Preserve the original trajectory time on the output axis.  Quenching
    # curves use time differences, so a nonzero starting time does not affect
    # the resulting lag-time curve.
    t_fast_ps = delta_t_fast_ps * np.arange(frame_start, frame_stop, dtype=np.float64)
    time_ps = t_fast_ps * kinetic_scale

    # Physical timestep for the interval hazard q(r) * dt.
    delta_t_physical_ps = delta_t_fast_ps * kinetic_scale
    dt_s = delta_t_physical_ps * 1.0e-12

    if eps is not None:
        x = (distances - a0) / eps
        soft_positive = eps * np.logaddexp(0.0, x)
        q_physical = q0 * np.exp(-beta * soft_positive)
    else:
        q_physical = q0 * np.exp(-beta * (distances - a0))

    decay_exponent = q_physical * dt_s

    return QuenchingDecayResult(
        time_ps=time_ps,
        decay_exponent_per_chain=np.asarray(decay_exponent, dtype=np.float64),
        chain_labels=labels,
        n_chains=int(n_chains),
        n_frames=int(n_frames),
        delta_t_fast_ps=float(delta_t_fast_ps),
        delta_t_physical_ps=float(delta_t_physical_ps),
        q0_per_s=float(q0),
        beta=float(beta),
        a0=float(a0),
        eps=None if eps is None else float(eps),
        kinetic_scale=float(kinetic_scale),
        distance_field=str(distance_field),
        source_frame_start=int(frame_start),
        source_frame_stop=int(frame_stop),
    )


def apply_decay(
    data: QuenchingInput,
    q0: float,
    beta: float,
    a0: float,
    *,
    delta_t_ps: float,
    eps: Optional[float] = None,
    kinetic_scale: float = 1.0,
    distance_field: str = "distance_per_chain_nm",
    time_start_ps: float = 0.0,
    time_stop_ps: Optional[float] = None,
) -> QuenchingOutput:
    """Apply the distance-dependent decay function to minimum-contact data.

    The standard input is either one ``MinimumContactDistanceResult`` or an
    arbitrarily keyed mapping of such results, for example the dictionary
    returned by ``condensate_analysis.get_values``. Nested mappings are handled
    recursively and their key structure is preserved; there is no assumption
    about replica numbers or a fixed number of data buckets.

    For every chain and frame, the function calculates::

        q(r) = q0 * exp[-beta * (r - a0)]
        exponent = q(r) * delta_t_physical_s

    When ``eps`` is supplied, ``r - a0`` is replaced by the smooth positive
    part used in the original implementation::

        eps * log(1 + exp((r - a0) / eps))

    Parameters
    ----------
    data
        One result containing ``distance_field`` with shape
        ``(n_chains, n_frames)``, or a possibly nested mapping of such results.
    q0
        Quenching rate in s^-1.
    beta
        Exponential distance coefficient in inverse distance units.
    a0
        Contact distance in the same units as the minimum-contact distances.
    delta_t_ps
        Spacing of the unscaled minimum-contact frames in picoseconds.
    eps
        Optional positive smoothing length.
    kinetic_scale
        Positive factor applied to both the output time axis and the physical
        timestep used in the hazard.
    distance_field
        Name of the two-dimensional distance field. The default matches
        ``MinimumContactDistanceResult``.
    time_start_ps, time_stop_ps
        Unscaled trajectory-time limits in picoseconds. Frames at
        ``t >= time_start_ps`` and ``t < time_stop_ps`` are retained. The stop
        limit is optional. Selection occurs before ``kinetic_scale`` is applied.

    Returns
    -------
    QuenchingDecayResult or dict
        One result for a single input object, or a mapping with the same nested
        key structure as ``data``.
    """
    q0_value = _finite_scalar(q0, name="q0")
    beta_value = _finite_scalar(beta, name="beta")
    a0_value = _finite_scalar(a0, name="a0")
    delta_t_value = _finite_scalar(delta_t_ps, name="delta_t_ps")
    kinetic_scale_value = _finite_scalar(kinetic_scale, name="kinetic_scale")
    time_start_value = _finite_scalar(time_start_ps, name="time_start_ps")
    if time_stop_ps is None:
        time_stop_value = None
    else:
        time_stop_value = _finite_scalar(time_stop_ps, name="time_stop_ps")

    if q0_value < 0.0:
        raise ValueError("q0 must be >= 0")
    if delta_t_value <= 0.0:
        raise ValueError("delta_t_ps must be > 0")
    if kinetic_scale_value <= 0.0:
        raise ValueError("kinetic_scale must be > 0")
    if time_start_value < 0.0:
        raise ValueError("time_start_ps must be >= 0")
    if time_stop_value is not None and time_stop_value <= time_start_value:
        raise ValueError("time_stop_ps must be greater than time_start_ps")
    if not isinstance(distance_field, str) or not distance_field.strip():
        raise ValueError("distance_field must be a non-empty string")

    if eps is None:
        eps_value = None
    else:
        eps_value = _finite_scalar(eps, name="eps")
        if eps_value <= 0.0:
            raise ValueError("eps must be > 0 when supplied")

    def transform(value: Any, path: str) -> QuenchingOutput:
        if _result_has_field(value, distance_field):
            return _apply_decay_to_result(
                value,
                q0=q0_value,
                beta=beta_value,
                a0=a0_value,
                delta_t_fast_ps=delta_t_value,
                eps=eps_value,
                kinetic_scale=kinetic_scale_value,
                distance_field=distance_field,
                time_start_fast_ps=time_start_value,
                time_stop_fast_ps=time_stop_value,
            )

        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"{path} is an empty mapping")
            return {key: transform(child, f"{path}[{key!r}]") for key, child in value.items()}

        raise TypeError(
            f"{path} must contain {distance_field!r} or be a mapping of " "compatible results"
        )

    return transform(data, "data")


def quench(arr: np.ndarray, tres: float = 100.0) -> np.ndarray:
    """Calculate one time-origin-averaged quenching survival curve.

    Parameters
    ----------
    arr
        Two-column array ``(time_ps, interval_hazard)``.  Hazard value ``k``
        applies to interval ``[time[k], time[k+1]]``; the final hazard value is
        therefore not integrated.
    tres
        Lag-time bin spacing in picoseconds.  Lag times are assigned to the
        nearest bin, reproducing the original implementation.

    Returns
    -------
    ndarray
        ``(n_populated_bins, 3)`` with columns ``lag_time_ps``, mean survival,
        and the number of time-origin pairs contributing to the bin.
    """
    tres_value = _finite_scalar(tres, name="tres")
    if tres_value <= 0.0:
        raise ValueError("tres must be > 0")

    values = np.asarray(arr, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"arr must have shape (n_frames, 2); got {values.shape}")

    t = values[:, 0]
    ex = values[:, 1]
    n = int(t.size)
    if n < 2:
        return np.zeros((0, 3), dtype=np.float64)
    if not np.all(np.isfinite(t)):
        raise ValueError("time values must be finite")
    if not np.all(np.diff(t) > 0.0):
        raise ValueError("time values must be strictly increasing")

    # ex[k] is the integrated hazard on [t[k], t[k+1]].
    ex_step = ex[:-1]

    # prefix[j] = sum(ex_step[0:j]), so the integrated hazard over [i,j] is
    # prefix[j] - prefix[i].
    prefix = np.empty(n, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(ex_step, out=prefix[1:])

    tmax = float(t[-1] - t[0])
    nmaxacc = int(tmax / tres_value + 1.5)
    if nmaxacc < 2:
        return np.zeros((0, 3), dtype=np.float64)

    acc = np.zeros(nmaxacc, dtype=np.float64)
    nacc = np.zeros(nmaxacc, dtype=np.int64)

    for i in range(n - 1):
        dt = t[i + 1 :] - t[i]
        bins = (dt / tres_value + 0.5).astype(np.int64)
        integrated_hazard = prefix[i + 1 :] - prefix[i]

        keep = (bins > 0) & (bins < nmaxacc)
        if np.any(keep):
            acc += np.bincount(
                bins[keep],
                weights=np.exp(-integrated_hazard[keep]),
                minlength=nmaxacc,
            )
            nacc += np.bincount(bins[keep], minlength=nmaxacc)

    populated = np.flatnonzero(nacc > 0)
    populated = populated[populated > 0]
    if populated.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    lag_time = populated.astype(np.float64) * tres_value
    mean_survival = acc[populated] / nacc[populated]
    counts = nacc[populated].astype(np.float64)
    return np.column_stack((lag_time, mean_survival, counts))


def _quench_decay_result(
    result: QuenchingDecayResult,
    *,
    tres: float,
) -> QuenchingCurveResult:
    hazards = np.asarray(result.decay_exponent_per_chain, dtype=np.float64)
    if hazards.ndim != 2:
        raise ValueError("decay_exponent_per_chain must have shape (n_chains, n_frames)")
    if hazards.shape != (int(result.n_chains), int(result.n_frames)):
        raise ValueError("decay_exponent_per_chain shape is inconsistent with result metadata")

    curves = [quench(np.column_stack((result.time_ps, row)), tres=tres) for row in hazards]
    nonempty = [curve for curve in curves if curve.shape[0] > 0]
    if not nonempty:
        return QuenchingCurveResult(
            lag_time_ps=np.zeros(0, dtype=np.float64),
            survival_per_chain=np.zeros((result.n_chains, 0), dtype=np.float64),
            counts_per_chain=np.zeros((result.n_chains, 0), dtype=np.float64),
            chain_labels=result.chain_labels,
            n_chains=int(result.n_chains),
            n_frames=int(result.n_frames),
            n_lag_bins=0,
            tres_ps=float(tres),
        )

    first_time = nonempty[0][:, 0]
    for chain_index, curve in enumerate(curves):
        if curve.shape[0] == 0:
            raise ValueError(f"chain {chain_index} produced no lag bins while other chains did")
        if curve.shape[0] != first_time.size or not np.array_equal(curve[:, 0], first_time):
            raise ValueError("chain quenching curves do not share lag-time bins")

    survival = np.stack([curve[:, 1] for curve in curves], axis=0)
    counts = np.stack([curve[:, 2] for curve in curves], axis=0)
    return QuenchingCurveResult(
        lag_time_ps=first_time.copy(),
        survival_per_chain=survival,
        counts_per_chain=counts,
        chain_labels=result.chain_labels,
        n_chains=int(result.n_chains),
        n_frames=int(result.n_frames),
        n_lag_bins=int(first_time.size),
        tres_ps=float(tres),
    )


def quench_all(
    data: Any,
    tres: float = 100.0,
) -> Union[QuenchingCurveResult, dict[Any, Any]]:
    """Calculate per-chain quenching curves for arbitrary result mappings.

    ``data`` may be one :class:`QuenchingDecayResult` or an arbitrarily nested
    mapping of such results.  Mapping keys and nesting are preserved; no fixed
    replica keys or number of buckets are assumed.
    """
    tres_value = _finite_scalar(tres, name="tres")
    if tres_value <= 0.0:
        raise ValueError("tres must be > 0")

    def transform(value: Any, path: str) -> Union[QuenchingCurveResult, dict[Any, Any]]:
        if isinstance(value, QuenchingDecayResult):
            return _quench_decay_result(value, tres=tres_value)

        # Duck-typed support for loaded or compatible result objects.
        if _result_has_field(value, "time_ps") and _result_has_field(
            value, "decay_exponent_per_chain"
        ):
            hazards = np.asarray(
                _result_field(value, "decay_exponent_per_chain"),
                dtype=np.float64,
            )
            if hazards.ndim != 2:
                raise ValueError(f"{path}.decay_exponent_per_chain must be two-dimensional")
            n_chains, n_frames = hazards.shape
            compatible = QuenchingDecayResult(
                time_ps=np.asarray(_result_field(value, "time_ps"), dtype=np.float64),
                decay_exponent_per_chain=hazards,
                chain_labels=_chain_labels(value, n_chains),
                n_chains=n_chains,
                n_frames=n_frames,
                delta_t_fast_ps=float(getattr(value, "delta_t_fast_ps", np.nan)),
                delta_t_physical_ps=float(getattr(value, "delta_t_physical_ps", np.nan)),
                q0_per_s=float(getattr(value, "q0_per_s", np.nan)),
                beta=float(getattr(value, "beta", np.nan)),
                a0=float(getattr(value, "a0", np.nan)),
                eps=getattr(value, "eps", None),
                kinetic_scale=float(getattr(value, "kinetic_scale", np.nan)),
                distance_field=str(getattr(value, "distance_field", "distance_per_chain_nm")),
            )
            if compatible.time_ps.shape != (n_frames,):
                raise ValueError(
                    f"{path}.time_ps has shape {compatible.time_ps.shape}, "
                    f"expected ({n_frames},)"
                )
            return _quench_decay_result(compatible, tres=tres_value)

        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"{path} is an empty mapping")
            return {key: transform(child, f"{path}[{key!r}]") for key, child in value.items()}

        raise TypeError(f"{path} must be a QuenchingDecayResult or a mapping of such results")

    return transform(data, "data")


__all__ = [
    "QuenchingCurveResult",
    "QuenchingDecayResult",
    "QuenchingInput",
    "QuenchingOutput",
    "apply_decay",
    "quench",
    "quench_all",
]
