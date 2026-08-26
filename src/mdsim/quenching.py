"""Distance-dependent Atto--Trp quenching analysis (2026-08-26 version).

This module operates on distance time series that have already been calculated
from molecular-dynamics trajectories. Its standard input is the
``MinimumContactDistanceResult`` produced by :mod:`mdsim.analysis`, but the
implementation uses duck typing so compatible result objects or mappings can
also be processed without importing the analysis module.

Two *different* trajectory observables are provided and must not be treated as
mathematically interchangeable:

* ``quench`` / ``quench_all`` calculate a first-quenching survival function
  from the integrated conditional hazard ``q[r(t)] dt``. This is an absorbing,
  renewal-style model: the first quenching event ends that observation window
  and all paths that have quenched receive zero subsequent weight. It is useful
  when adopting the reaction--diffusion/first-passage interpretation of the
  Atto--Trp experiment.
* ``fluorescence_autocorrelation`` calculates the equilibrium autocorrelation
  of a simulated fluorescence-brightness signal. This is the more direct
  analogue of what FCS measures and allows quenching/rebrightening cycles
  between the two endpoints of a correlation interval.

The two decays can share a characteristic time in a diffusion-limited,
fast-reset limit, but they generally differ when recovery/dissociation and
rebinding are important. The approximately 2 ns Atto singlet lifetime is
coarse-grained out of the microsecond first-quenching survival calculation; it
enters only the optional instantaneous dynamic-yield brightness mapping.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional, Union

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
    cap_rate_at_q0: bool = False

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
    """Time-origin-averaged first-quenching survival curves.

    ``survival_per_chain`` and ``counts_per_chain`` have shape
    ``(n_chains, n_lag_bins)``. The values reproduce :func:`quench` applied
    independently to every chain in a :class:`QuenchingDecayResult`.

    This object is an absorbing first-event observable, not an equilibrium
    fluorescence autocorrelation. A path contributes at lag ``tau`` only when
    no quenching event has occurred since its selected time origin.
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


QuenchingInput = Union[Any, Mapping[Any, "QuenchingInput"]]
QuenchingOutput = Union[
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
    cap_rate_at_q0: bool,
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

    q_physical = _distance_dependent_quenching_rate(
        distances,
        q0=q0,
        beta=beta,
        a0=a0,
        eps=eps,
        cap_rate_at_q0=cap_rate_at_q0,
    )

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
        cap_rate_at_q0=bool(cap_rate_at_q0),
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
    cap_rate_at_q0: bool = False,
) -> QuenchingOutput:
    """Apply the distance-dependent first-quenching hazard to distance data.

    Conditional on the supplied trajectory, quenching is assumed to be a
    memoryless Poisson event with instantaneous rate ``q[r(t)]``. The resulting
    interval hazards are intended for an absorbing first-event survival model:
    once an event occurs, that time-origin trajectory no longer contributes.
    Normal fluorescence decay, re-excitation, dark-state recovery, and
    quenching-induced changes to the conformational trajectory are not included.

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
    cap_rate_at_q0
        If True and ``eps`` is None, cap the rate at ``q0`` for distances below
        ``a0``. The default is False for backward compatibility with earlier
        caches and calculations.

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
    if beta_value < 0.0:
        raise ValueError("beta must be >= 0")
    if not isinstance(cap_rate_at_q0, (bool, np.bool_)):
        raise TypeError("cap_rate_at_q0 must be a boolean")
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
                cap_rate_at_q0=bool(cap_rate_at_q0),
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
    """Calculate one time-origin-averaged first-quenching survival curve.

    For a time origin ``t0`` and lag ``tau`` this evaluates

    ``S(tau|t0) = exp[-integral(t0,t0+tau) q(r(t)) dt]``

    and averages over time origins. The construction assumes that ``q(r)`` is
    a conditional Poisson hazard and that the first quenching event is an
    absorbing event which completely erases the correlation memory for that
    observation window. Rebrightening or dissociation followed by renewed
    emission inside the same interval is therefore excluded. This is not, in
    general, the same quantity as a fluorescence-intensity autocorrelation.

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
                cap_rate_at_q0=bool(getattr(value, "cap_rate_at_q0", False)),
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


# --- FCS-like fluorescence signal and autocorrelation -------------------------


@dataclass(frozen=True)
class FluorescenceSignalResult:
    """Relative fluorescence brightness inferred from a distance trajectory.

    In the ``dynamic-yield`` approximation, the quencher adds the
    distance-dependent excited-state decay rate ``q(r)`` to the intrinsic Atto
    decay rate ``k_intrinsic = 1/tau_intrinsic``. Assuming the radiative rate is
    unchanged and the molecular geometry is effectively frozen during one
    approximately 2 ns optical cycle, the fluorescence brightness relative to
    unquenched Atto is

    ``brightness(r) = k_intrinsic / (k_intrinsic + q(r))``.

    This mapping coarse-grains rapid repeated excitation/emission cycles and is
    suitable for constructing an equilibrium FCS-like intensity signal. It
    does not explicitly model a persistent nonfluorescent ground-state
    Atto--Trp complex; such a state would require an additional association /
    dissociation kinetic model.

    ``brightness_per_chain`` has shape ``(n_chains, n_frames)``.  This is an
    equilibrium fluorescence observable for constructing an FCS-like intensity
    autocorrelation; it is distinct from :class:`QuenchingDecayResult`, whose
    interval hazards are integrated to make an irreversible survival curve.
    """

    time_ps: np.ndarray  # (n_frames,), physical time after kinetic scaling
    brightness_per_chain: np.ndarray  # (n_chains, n_frames), relative brightness 0..1
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
    intrinsic_lifetime_ps: float
    intrinsic_rate_per_s: float
    distance_field: str = "distance_per_chain_nm"
    source_frame_start: int = 0
    source_frame_stop: Optional[int] = None
    cap_rate_at_q0: bool = True

    @property
    def relative_brightness_per_chain(self) -> np.ndarray:
        """Alias for :attr:`brightness_per_chain`."""
        return self.brightness_per_chain

    @property
    def signal_per_chain(self) -> np.ndarray:
        """Alias emphasizing use as the simulated fluorescence signal."""
        return self.brightness_per_chain

    @property
    def quenched_fraction_per_chain(self) -> np.ndarray:
        """Fractional loss of fluorescence, ``1 - relative brightness``."""
        return 1.0 - np.asarray(self.brightness_per_chain, dtype=np.float64)

    @property
    def quenching_rate_per_chain_s(self) -> np.ndarray:
        """Recover ``q(r)`` from the stored relative brightness.

        This is calculated on access to avoid storing a second trajectory-sized
        array in the cache.  Exactly zero brightness maps to ``+inf``.
        """
        brightness = np.asarray(self.brightness_per_chain, dtype=np.float64)
        rate = np.full_like(brightness, np.nan, dtype=np.float64)
        finite = np.isfinite(brightness)
        positive = finite & (brightness > 0.0)
        rate[positive] = float(self.intrinsic_rate_per_s) * (
            (1.0 - brightness[positive]) / brightness[positive]
        )
        rate[finite & (brightness == 0.0)] = np.inf
        return rate

    @property
    def series(self) -> tuple[np.ndarray, ...]:
        """Return one ``(time_ps, relative_brightness)`` array per chain."""
        return tuple(np.column_stack((self.time_ps, row)) for row in self.brightness_per_chain)


@dataclass(frozen=True)
class FluorescenceAutocorrelationResult:
    """Per-chain autocorrelation of a simulated fluorescence signal.

    For ``normalization='fcs'`` the returned curve is

    ``<delta I(t) delta I(t+lag)> / <I>**2``,

    matching the internal-state contribution to a normalized FCS correlation.
    ``normalization='coefficient'`` instead divides by the zero-lag variance so
    that the curve begins at one, and ``'covariance'`` leaves the covariance in
    signal-squared units.

    The ``survival_per_chain`` property is a compatibility alias that permits
    use with the existing notebook-level ``calculate_quenching_statistics`` and
    ``fit_quenching_decay`` helpers.  For this result it represents an
    autocorrelation, not a survival probability.
    """

    lag_time_ps: np.ndarray  # (n_lag_bins,)
    correlation_per_chain: np.ndarray  # (n_chains, n_lag_bins)
    counts_per_chain: np.ndarray  # (n_chains, n_lag_bins)
    mean_signal_per_chain: np.ndarray  # (n_chains,)
    variance_signal_per_chain: np.ndarray  # (n_chains,)
    chain_labels: tuple[str, ...]
    lags_frames: np.ndarray  # (n_lag_bins,)
    n_chains: int
    n_frames: int
    n_lag_bins: int
    delta_t_physical_ps: float
    normalization: str
    signal_field: str
    unbiased: bool
    min_pairs: int

    @property
    def time_ps(self) -> np.ndarray:
        """Alias for :attr:`lag_time_ps`."""
        return self.lag_time_ps

    @property
    def curve_per_chain(self) -> np.ndarray:
        """Alias for :attr:`correlation_per_chain`."""
        return self.correlation_per_chain

    @property
    def quenching_curve_per_chain(self) -> np.ndarray:
        """Compatibility alias for :attr:`correlation_per_chain`."""
        return self.correlation_per_chain

    @property
    def survival_per_chain(self) -> np.ndarray:
        """Compatibility alias used by existing notebook aggregation code.

        The values are fluorescence autocorrelations, not survival probabilities.
        """
        return self.correlation_per_chain

    @property
    def series(self) -> tuple[np.ndarray, ...]:
        """Return one ``(lag_time_ps, correlation, count)`` array per chain."""
        return tuple(
            np.column_stack((self.lag_time_ps, values, counts))
            for values, counts in zip(self.correlation_per_chain, self.counts_per_chain)
        )


FluorescenceSignalOutput = Union[
    FluorescenceSignalResult,
    dict[Any, "FluorescenceSignalOutput"],
]
FluorescenceCorrelationOutput = Union[
    FluorescenceAutocorrelationResult,
    dict[Any, "FluorescenceCorrelationOutput"],
]


def _distance_dependent_quenching_rate(
    distances: np.ndarray,
    *,
    q0: float,
    beta: float,
    a0: float,
    eps: Optional[float],
    cap_rate_at_q0: bool,
) -> np.ndarray:
    """Evaluate ``q(r)=q0*exp[-beta*(r-a0)]`` with optional contact capping.

    When ``eps`` is supplied, the same smooth positive-part convention as
    :func:`apply_decay` is used.  Otherwise ``cap_rate_at_q0=True`` replaces
    ``r-a0`` by ``max(r-a0, 0)``, respecting the interpretation of ``q0`` as
    the rate at closest approach.  Set it to False for exact backward-style
    extrapolation below ``a0``.
    """
    distance = np.asarray(distances, dtype=np.float64)
    if eps is not None:
        x = (distance - float(a0)) / float(eps)
        separation = float(eps) * np.logaddexp(0.0, x)
    elif bool(cap_rate_at_q0):
        separation = np.maximum(distance - float(a0), 0.0)
    else:
        separation = distance - float(a0)

    if float(q0) == 0.0:
        return np.zeros_like(distance, dtype=np.float64)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        rate = float(q0) * np.exp(-float(beta) * separation)
    return np.asarray(rate, dtype=np.float64)


def _apply_fluorescence_signal_to_result(
    result: Any,
    *,
    q0: float,
    beta: float,
    a0: float,
    intrinsic_lifetime_ps: float,
    delta_t_fast_ps: float,
    eps: Optional[float],
    kinetic_scale: float,
    distance_field: str,
    time_start_fast_ps: float,
    time_stop_fast_ps: Optional[float],
    cap_rate_at_q0: bool,
) -> FluorescenceSignalResult:
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
        raise ValueError(
            "the requested time range must contain at least two frames for autocorrelation"
        )

    labels = _chain_labels(result, n_chains)
    t_fast_ps = delta_t_fast_ps * np.arange(frame_start, frame_stop, dtype=np.float64)
    time_ps = t_fast_ps * kinetic_scale
    delta_t_physical_ps = delta_t_fast_ps * kinetic_scale

    q_physical = _distance_dependent_quenching_rate(
        distances,
        q0=q0,
        beta=beta,
        a0=a0,
        eps=eps,
        cap_rate_at_q0=cap_rate_at_q0,
    )
    intrinsic_rate = 1.0 / (float(intrinsic_lifetime_ps) * 1.0e-12)

    # Relative quantum yield/brightness under an added non-radiative channel q(r).
    with np.errstate(divide="ignore", invalid="ignore"):
        brightness = intrinsic_rate / (intrinsic_rate + q_physical)
    brightness = np.asarray(brightness, dtype=np.float64)

    return FluorescenceSignalResult(
        time_ps=time_ps,
        brightness_per_chain=brightness,
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
        intrinsic_lifetime_ps=float(intrinsic_lifetime_ps),
        intrinsic_rate_per_s=float(intrinsic_rate),
        distance_field=str(distance_field),
        source_frame_start=int(frame_start),
        source_frame_stop=int(frame_stop),
        cap_rate_at_q0=bool(cap_rate_at_q0),
    )


def apply_fluorescence_signal(
    data: QuenchingInput,
    q0: float,
    beta: float,
    a0: float,
    *,
    intrinsic_lifetime_ps: float,
    delta_t_ps: float,
    eps: Optional[float] = None,
    kinetic_scale: float = 1.0,
    distance_field: str = "distance_per_chain_nm",
    time_start_ps: float = 0.0,
    time_stop_ps: Optional[float] = None,
    cap_rate_at_q0: bool = True,
) -> FluorescenceSignalOutput:
    """Map distance trajectories to relative Atto fluorescence brightness.

    For every chain and frame this function evaluates

    ``q(r) = q0 * exp[-beta * (r-a0)]``

    and then

    ``B(r) = k_intrinsic / (k_intrinsic + q(r))``,

    where ``k_intrinsic = 1/intrinsic_lifetime``.  No constant is subtracted
    from ``q(r)``.  The intrinsic lifetime enters only through this competition
    between ordinary excited-state decay and Trp-induced electron transfer.

    ``kinetic_scale`` changes the physical time axis used by the later
    autocorrelation but does not change the instantaneous brightness mapping.
    Numeric distance parameters must use the same distance unit as
    ``distance_field`` (normally nm, so a beta reported in Angstrom^-1 must be
    multiplied by ten).
    """
    q0_value = _finite_scalar(q0, name="q0")
    beta_value = _finite_scalar(beta, name="beta")
    a0_value = _finite_scalar(a0, name="a0")
    lifetime_value = _finite_scalar(intrinsic_lifetime_ps, name="intrinsic_lifetime_ps")
    delta_t_value = _finite_scalar(delta_t_ps, name="delta_t_ps")
    kinetic_scale_value = _finite_scalar(kinetic_scale, name="kinetic_scale")
    time_start_value = _finite_scalar(time_start_ps, name="time_start_ps")
    time_stop_value = (
        None if time_stop_ps is None else _finite_scalar(time_stop_ps, name="time_stop_ps")
    )

    if q0_value < 0.0:
        raise ValueError("q0 must be >= 0")
    if beta_value < 0.0:
        raise ValueError("beta must be >= 0")
    if lifetime_value <= 0.0:
        raise ValueError("intrinsic_lifetime_ps must be > 0")
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
    if not isinstance(cap_rate_at_q0, (bool, np.bool_)):
        raise TypeError("cap_rate_at_q0 must be a boolean")

    if eps is None:
        eps_value = None
    else:
        eps_value = _finite_scalar(eps, name="eps")
        if eps_value <= 0.0:
            raise ValueError("eps must be > 0 when supplied")

    def transform(value: Any, path: str) -> FluorescenceSignalOutput:
        if _result_has_field(value, distance_field):
            return _apply_fluorescence_signal_to_result(
                value,
                q0=q0_value,
                beta=beta_value,
                a0=a0_value,
                intrinsic_lifetime_ps=lifetime_value,
                delta_t_fast_ps=delta_t_value,
                eps=eps_value,
                kinetic_scale=kinetic_scale_value,
                distance_field=distance_field,
                time_start_fast_ps=time_start_value,
                time_stop_fast_ps=time_stop_value,
                cap_rate_at_q0=bool(cap_rate_at_q0),
            )
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"{path} is an empty mapping")
            return {key: transform(child, f"{path}[{key!r}]") for key, child in value.items()}
        raise TypeError(
            f"{path} must contain {distance_field!r} or be a mapping of compatible results"
        )

    return transform(data, "data")


def _normalize_fcs_normalization(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("normalization must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fcs": "fcs",
        "mean_squared": "fcs",
        "mean2": "fcs",
        "coefficient": "coefficient",
        "normalized": "coefficient",
        "correlation_coefficient": "coefficient",
        "unit": "coefficient",
        "covariance": "covariance",
        "cov": "covariance",
        "none": "covariance",
    }
    if normalized not in aliases:
        raise ValueError("normalization must be 'fcs', 'coefficient', or 'covariance'")
    return aliases[normalized]


def _regular_timestep_ps(time_ps: np.ndarray) -> float:
    time = np.asarray(time_ps, dtype=np.float64).reshape(-1)
    if time.size < 2:
        raise ValueError("time_ps must contain at least two values")
    if not np.all(np.isfinite(time)):
        raise ValueError("time_ps must be finite")
    steps = np.diff(time)
    if np.any(steps <= 0.0):
        raise ValueError("time_ps must be strictly increasing")
    dt = float(np.median(steps))
    tolerance = max(1.0e-9, abs(dt) * 1.0e-8)
    if not np.allclose(steps, dt, rtol=1.0e-8, atol=tolerance):
        raise ValueError("fluorescence autocorrelation requires regularly spaced time samples")
    return dt


def _fft_autocovariance_one(
    values: np.ndarray,
    *,
    max_lag_frames: int,
    unbiased: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Autocovariance and finite-pair counts for one possibly gapped series."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(x.size)
    if n < 2:
        raise ValueError("signal series must contain at least two frames")
    max_lag = int(max_lag_frames)
    if max_lag < 0 or max_lag >= n:
        raise ValueError(f"max_lag_frames must be in 0..{n - 1}")

    finite = np.isfinite(x)
    n_finite = int(np.sum(finite))
    if n_finite == 0:
        return (
            np.full(max_lag + 1, np.nan, dtype=np.float64),
            np.zeros(max_lag + 1, dtype=np.int64),
            float("nan"),
            float("nan"),
        )

    mean = float(np.mean(x[finite]))
    centered = np.where(finite, x - mean, 0.0)
    mask = finite.astype(np.float64)

    nfft = 1 << int((2 * n - 1).bit_length())
    spectrum = np.fft.rfft(centered, n=nfft)
    numerator = np.fft.irfft(spectrum * np.conjugate(spectrum), n=nfft)[: max_lag + 1]

    mask_spectrum = np.fft.rfft(mask, n=nfft)
    count_float = np.fft.irfft(mask_spectrum * np.conjugate(mask_spectrum), n=nfft)[: max_lag + 1]
    counts = np.rint(np.maximum(count_float, 0.0)).astype(np.int64)

    covariance = np.full(max_lag + 1, np.nan, dtype=np.float64)
    if bool(unbiased):
        np.divide(numerator, counts, out=covariance, where=counts > 0)
    else:
        covariance[counts > 0] = numerator[counts > 0] / float(n_finite)

    variance = float(covariance[0]) if counts[0] > 0 else float("nan")
    return covariance, counts, mean, variance


def fluorescence_autocorrelation(
    result: Any,
    *,
    signal_field: str = "brightness_per_chain",
    normalization: str = "fcs",
    max_lag_ps: Optional[float] = None,
    max_lag_frames: Optional[int] = None,
    lag_spacing_ps: Optional[float] = None,
    lag_stride: int = 1,
    include_zero_lag: bool = False,
    unbiased: bool = True,
    min_pairs: int = 2,
) -> FluorescenceAutocorrelationResult:
    """Calculate per-chain fluorescence autocorrelations using FFTs.

    The default ``normalization='fcs'`` returns

    ``G_internal(lag) = <delta B(t) delta B(t+lag)> / <B>**2``.

    This quantity can be compared with the fitted intramolecular term
    ``A_I exp(-lag/tau_I)`` after translational diffusion and dye-specific
    photophysical components have been separated experimentally.

    ``lag_spacing_ps`` thins the regularly sampled correlation to the nearest
    integer number of trajectory frames. It is mutually exclusive with a
    non-default ``lag_stride``. ``max_lag_ps`` and ``max_lag_frames`` are also
    mutually exclusive.
    """
    if not isinstance(signal_field, str) or not signal_field.strip():
        raise ValueError("signal_field must be a non-empty string")
    norm = _normalize_fcs_normalization(normalization)
    if not isinstance(include_zero_lag, (bool, np.bool_)):
        raise TypeError("include_zero_lag must be a boolean")
    if not isinstance(unbiased, (bool, np.bool_)):
        raise TypeError("unbiased must be a boolean")
    if isinstance(min_pairs, (bool, np.bool_)):
        raise TypeError("min_pairs must be an integer >= 1")
    min_pairs_i = int(min_pairs)
    if min_pairs_i < 1 or float(min_pairs_i) != float(min_pairs):
        raise ValueError("min_pairs must be an integer >= 1")

    signal = np.asarray(_result_field(result, signal_field), dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(
            f"{signal_field!r} must have shape (n_chains, n_frames); got {signal.shape}"
        )
    n_chains, n_frames = signal.shape
    if n_chains < 1 or n_frames < 2:
        raise ValueError("signal must contain at least one chain and two frames")

    time_ps = np.asarray(_result_field(result, "time_ps"), dtype=np.float64)
    if time_ps.shape != (n_frames,):
        raise ValueError(f"time_ps has shape {time_ps.shape}, expected ({n_frames},)")
    dt_ps = _regular_timestep_ps(time_ps)

    if max_lag_ps is not None and max_lag_frames is not None:
        raise ValueError("max_lag_ps and max_lag_frames are mutually exclusive")
    if max_lag_ps is None:
        max_lag = n_frames - 1 if max_lag_frames is None else int(max_lag_frames)
    else:
        max_lag_value = _finite_scalar(max_lag_ps, name="max_lag_ps")
        if max_lag_value <= 0.0:
            raise ValueError("max_lag_ps must be > 0")
        max_lag = int(math.floor(max_lag_value / dt_ps + 1.0e-12))
    if max_lag < 0:
        raise ValueError("max_lag_frames must be >= 0")
    max_lag = min(int(max_lag), n_frames - 1)

    if isinstance(lag_stride, (bool, np.bool_)):
        raise TypeError("lag_stride must be an integer >= 1")
    stride = int(lag_stride)
    if stride < 1 or float(stride) != float(lag_stride):
        raise ValueError("lag_stride must be an integer >= 1")

    if lag_spacing_ps is not None:
        if stride != 1:
            raise ValueError("lag_spacing_ps and a non-default lag_stride are mutually exclusive")
        spacing = _finite_scalar(lag_spacing_ps, name="lag_spacing_ps")
        if spacing <= 0.0:
            raise ValueError("lag_spacing_ps must be > 0")
        stride = max(1, int(round(spacing / dt_ps)))

    start_lag = 0 if bool(include_zero_lag) else stride
    if max_lag < start_lag:
        raise ValueError(
            "selected maximum lag is shorter than the first output lag; "
            "increase max_lag or include zero lag"
        )
    lags = np.arange(start_lag, max_lag + 1, stride, dtype=np.int64)

    correlation = np.full((n_chains, lags.size), np.nan, dtype=np.float64)
    counts_out = np.zeros((n_chains, lags.size), dtype=np.int64)
    means = np.full(n_chains, np.nan, dtype=np.float64)
    variances = np.full(n_chains, np.nan, dtype=np.float64)

    for chain_index in range(n_chains):
        covariance, counts, mean, variance = _fft_autocovariance_one(
            signal[chain_index],
            max_lag_frames=max_lag,
            unbiased=bool(unbiased),
        )
        selected_covariance = covariance[lags]
        selected_counts = counts[lags]

        if norm == "fcs":
            denominator = mean * mean
        elif norm == "coefficient":
            denominator = variance
        else:
            denominator = 1.0

        if np.isfinite(denominator) and denominator > 0.0:
            selected = selected_covariance / denominator
        elif norm == "covariance":
            selected = selected_covariance
        else:
            selected = np.full_like(selected_covariance, np.nan)

        selected = np.asarray(selected, dtype=np.float64)
        selected[selected_counts < min_pairs_i] = np.nan
        correlation[chain_index, :] = selected
        counts_out[chain_index, :] = selected_counts
        means[chain_index] = mean
        variances[chain_index] = variance

    labels = _chain_labels(result, n_chains)
    return FluorescenceAutocorrelationResult(
        lag_time_ps=lags.astype(np.float64) * dt_ps,
        correlation_per_chain=correlation,
        counts_per_chain=counts_out.astype(np.float64),
        mean_signal_per_chain=means,
        variance_signal_per_chain=variances,
        chain_labels=labels,
        lags_frames=lags,
        n_chains=int(n_chains),
        n_frames=int(n_frames),
        n_lag_bins=int(lags.size),
        delta_t_physical_ps=float(dt_ps),
        normalization=norm,
        signal_field=str(signal_field),
        unbiased=bool(unbiased),
        min_pairs=int(min_pairs_i),
    )


def fluorescence_autocorrelation_all(
    data: Any,
    *,
    signal_field: str = "brightness_per_chain",
    normalization: str = "fcs",
    max_lag_ps: Optional[float] = None,
    max_lag_frames: Optional[int] = None,
    lag_spacing_ps: Optional[float] = None,
    lag_stride: int = 1,
    include_zero_lag: bool = False,
    unbiased: bool = True,
    min_pairs: int = 2,
) -> FluorescenceCorrelationOutput:
    """Recursively calculate FCS-like autocorrelations for signal results."""

    def transform(value: Any, path: str) -> FluorescenceCorrelationOutput:
        if _result_has_field(value, signal_field) and _result_has_field(value, "time_ps"):
            return fluorescence_autocorrelation(
                value,
                signal_field=signal_field,
                normalization=normalization,
                max_lag_ps=max_lag_ps,
                max_lag_frames=max_lag_frames,
                lag_spacing_ps=lag_spacing_ps,
                lag_stride=lag_stride,
                include_zero_lag=include_zero_lag,
                unbiased=unbiased,
                min_pairs=min_pairs,
            )
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(f"{path} is an empty mapping")
            return {key: transform(child, f"{path}[{key!r}]") for key, child in value.items()}
        raise TypeError(
            f"{path} must contain {signal_field!r} and 'time_ps', or be a mapping "
            "of compatible fluorescence-signal results"
        )

    return transform(data, "data")


# Explicit names that distinguish the absorbing first-event observable from FCS.
first_quench_survival = quench
first_quench_survival_all = quench_all

# Concise aliases for notebook code that prefers explicit FCS terminology.
apply_fcs_signal = apply_fluorescence_signal
fcs_autocorrelation = fluorescence_autocorrelation
fcs_autocorrelation_all = fluorescence_autocorrelation_all


__all__ = [
    "FluorescenceAutocorrelationResult",
    "FluorescenceCorrelationOutput",
    "FluorescenceSignalOutput",
    "FluorescenceSignalResult",
    "QuenchingCurveResult",
    "QuenchingDecayResult",
    "QuenchingInput",
    "QuenchingOutput",
    "apply_decay",
    "apply_fcs_signal",
    "apply_fluorescence_signal",
    "fcs_autocorrelation",
    "fcs_autocorrelation_all",
    "first_quench_survival",
    "first_quench_survival_all",
    "fluorescence_autocorrelation",
    "fluorescence_autocorrelation_all",
    "quench",
    "quench_all",
]
