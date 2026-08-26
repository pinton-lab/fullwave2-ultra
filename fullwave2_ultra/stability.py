"""Time-step stability (the CFL condition) for the shipped solver binaries.

Two separate things are easy to get wrong, and the second one is what actually
bites in heterogeneous media.

**The limit is not 1/sqrt(dim).** The staggered leapfrog these solvers use has
the dispersion relation ``sin(w*dT/2) = r * Sbar(k*dX, n)`` with
``r = c*dT/dX``, so ``w`` stays real -- the scheme stays stable -- only while
``r*Sbar <= 1`` over the whole Brillouin zone and every direction ``n``. The
optimized (dispersion-relation-preserving) tap sets carry MORE zone-corner
amplitude than a plain second-order stencil, which is the price of their phase
accuracy, so they are LESS stable than the textbook ``1/sqrt(dim)`` value. Use
the measured numbers below, never the folklore one.

**``r`` is local; ``cfl`` is nominal.** The deck writers take a ``cfl`` and set
``dT = dX/c0*cfl``, referencing it to ``c0``. Stability is set by the FASTEST
material in the map, so the ratio the scheme actually runs at is

    r_realized = cfl * max(c) / c0

With ``c0 = 1540`` and cortical bone at 2900 m/s, a requested ``cfl = 0.30`` is
really 0.565. Such a run is clean until the wave reaches the bone and then goes
to NaN across the whole field. Nothing about the requested number looks wrong;
only the realized one does, which is why :func:`check_cfl` reports it.

Note this package publishes the 2D limits only. The 3D tap sets are not part of
the public package, so :func:`cfl_limit` has no 3D entry and
:func:`check_cfl` on a 3D deck reports the realized ratio without comparing it
to a threshold. The reference-speed warning is dimension-independent and works
in both.
"""
from __future__ import annotations

import warnings

#: Largest stable Courant ratio ``r = c*dT/dX``, keyed by ``(M, dim)``.
#: Measured by bisection on ``r`` against ``max(r*Sbar) <= 1`` over the
#: Brillouin zone, converged to 1e-6 across a 50x refinement of the wavenumber
#: grid; the worst direction is the main diagonal. Compare with
#: ``1/sqrt(2) = 0.7071`` to see how much the phase optimization costs.
CFL_LIMIT = {
    (8, 2): 0.5326,
    (6, 2): 0.6108,
}


def cfl_limit(*, M: int = 8, dim: int = 2) -> float:
    """Largest stable ``r = c*dT/dX`` for this tap set.

    This bounds the LOCAL ratio in the FASTEST material of the map, not the
    nominal ``cfl`` a deck writer is handed. Raises for combinations this
    package does not publish.
    """
    key = (int(M), int(dim))
    if key not in CFL_LIMIT:
        raise KeyError(
            f"no published CFL limit for M={M}, dim={dim} (have "
            f"{sorted(CFL_LIMIT)}). The 3D tap sets are not part of the public "
            "package; size a 3D time step conservatively and watch for NaN.")
    return CFL_LIMIT[key]


def realized_cfl(c_max: float, dT: float, dX: float) -> float:
    """The Courant ratio the scheme actually runs at: ``max(c)*dT/dX``."""
    return float(c_max) * float(dT) / float(dX)


def check_cfl(c_max: float, dT: float, dX: float, *, c0: float | None = None,
              M: int = 8, dim: int = 2, warn_frac: float = 0.90,
              ref_tol: float = 0.20, on_violation: str = "raise") -> dict:
    """Check the realized Courant ratio against the published limit.

    ``c_max`` is the fastest sound speed in the map (extended grid). Pass ``c0``
    and the reference-speed explanation is folded into whatever this reports.

    Where a limit is published (2D), that comparison is authoritative: over it
    raises, near it warns. Where none is (3D), the best available proxy is the
    gap between what you asked for and what the scheme will run at, so this
    warns when the realized ratio exceeds the nominal one by more than
    ``ref_tol``. That tolerance exists so ordinary tissue heterogeneity -- a few
    percent of scatterer contrast above ``c0`` -- stays silent; the case worth
    hearing about is bone at nearly twice ``c0``.

    ``on_violation``: ``"raise"`` (default), ``"warn"`` or ``"ignore"``.
    Returns ``dict(r, limit, frac, ok)``; ``limit``/``frac``/``ok`` are ``None``
    when this package publishes no limit for the combination.
    """
    r = realized_cfl(c_max, dT, dX)
    lim = CFL_LIMIT.get((int(M), int(dim)))
    out = dict(r=r, limit=lim, frac=(r / lim if lim else None),
               ok=(r <= lim if lim else None))
    if on_violation == "ignore":
        return out

    nominal = None
    if c0 is not None and float(c0) > 0 and float(c_max) > 0:
        nominal = r * float(c0) / float(c_max)
    inflated = nominal is not None and r > nominal * (1.0 + ref_tol)
    why = ""
    if inflated:
        why = (f" You asked for cfl={nominal:.4f} against c0={float(c0):.1f} m/s, "
               f"but the fastest material is {float(c_max):.1f} m/s, so the "
               f"scheme runs at {r:.4f}. Size dT from max(c), not c0.")

    if lim is None:
        if inflated:
            warnings.warn(
                f"no CFL limit is published for {dim}D in this package, so the "
                f"time step cannot be checked against one.{why}", stacklevel=2)
        return out

    if r > lim:
        msg = (f"unstable time step: the local Courant ratio in the fastest "
               f"material is r = {r:.4f}, above the measured {dim}D M{M} limit "
               f"{lim:.4f} ({r / lim:.2f}x). The scheme will diverge once the "
               f"wave reaches that material. Reduce dT by at least "
               f"{r / lim:.3f}x.{why}")
        if on_violation == "raise":
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    elif r / lim > warn_frac:
        warnings.warn(
            f"time step is within {100 * (1 - r / lim):.1f}% of the {dim}D M{M} "
            f"stability limit (r={r:.4f}, limit={lim:.4f}). Heterogeneity, "
            f"nonlinearity and float32 round-off all eat into that margin.{why}",
            stacklevel=2)
    return out
