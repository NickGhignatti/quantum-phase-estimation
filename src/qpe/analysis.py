"""Turning QPE measurement counts into phases, and plotting them.

The conversion is deliberately trivial (see :mod:`qpe.qft` for why no bit reversal is
needed) but it lives in one place so that notebooks, tests and the HHL port all read
results the same way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "counts_to_phases",
    "best_phase",
    "PhaseEstimate",
    "expected_phase_distribution",
    "plot_phase_distribution",
    "plot_stage_comparison",
]


@dataclass(frozen=True)
class PhaseEstimate:
    """The outcome of reading a QPE register."""

    phase: float
    """Most likely phase, in [0, 1)."""
    probability: float
    """Probability of the most likely outcome."""
    distribution: dict[float, float]
    """Full phase -> probability distribution."""
    num_eval_qubits: int

    @property
    def resolution(self) -> float:
        """Spacing of representable phases, ``2**-n``."""
        return 2.0**-self.num_eval_qubits

    def error_vs(self, true_phase: float) -> float:
        """Absolute error against a known phase, accounting for wraparound at 1.0."""
        d = abs(self.phase - true_phase) % 1.0
        return min(d, 1.0 - d)

    def mean_phase(self) -> float:
        """Probability-weighted circular mean, robust to the wrap at 0/1.

        Useful for non-dyadic phases, where the distribution straddles two neighbouring
        representable values and the modal estimate alone throws away information.
        """
        angles = np.array([2 * np.pi * p for p in self.distribution])
        weights = np.array(list(self.distribution.values()))
        z = np.sum(weights * np.exp(1j * angles))
        return float((np.angle(z) / (2 * np.pi)) % 1.0)


def counts_to_phases(counts: dict[str, int], num_eval_qubits: int) -> dict[float, float]:
    """Convert a counts dict to a ``phase -> probability`` mapping."""
    total = sum(counts.values())
    if total == 0:
        return {}
    scale = 2.0**num_eval_qubits
    out: dict[float, float] = {}
    for bitstring, count in counts.items():
        # Tolerate multi-register keys ("010 1") by taking the phase register, which
        # qpe_circuit always places first.
        token = bitstring.split()[0]
        phase = int(token, 2) / scale
        out[phase] = out.get(phase, 0.0) + count / total
    return out


def best_phase(counts: dict[str, int], num_eval_qubits: int) -> PhaseEstimate:
    """Read the most likely phase out of QPE counts."""
    dist = counts_to_phases(counts, num_eval_qubits)
    if not dist:
        raise ValueError("Empty counts; nothing to estimate.")
    phase = max(dist, key=dist.__getitem__)
    return PhaseEstimate(
        phase=phase,
        probability=dist[phase],
        distribution=dist,
        num_eval_qubits=num_eval_qubits,
    )


def expected_phase_distribution(true_phase: float, num_eval_qubits: int) -> dict[float, float]:
    """Analytic QPE outcome distribution for an exact eigenstate.

    The textbook result: measuring outcome ``m`` on ``n`` qubits has probability

        P(m) = |sin(pi * 2**n * (theta - m/2**n)) / (2**n * sin(pi * (theta - m/2**n)))|**2

    reducing to a Kronecker delta when ``theta`` is an exact multiple of ``2**-n``.
    Notebooks use this to overlay theory on sampled histograms.
    """
    n = num_eval_qubits
    N = 2**n
    out: dict[float, float] = {}
    for m in range(N):
        delta = true_phase - m / N
        if abs(delta % 1.0) < 1e-12 or abs(delta % 1.0 - 1.0) < 1e-12:
            prob = 1.0
        else:
            num = np.sin(np.pi * N * delta)
            den = N * np.sin(np.pi * delta)
            prob = float(abs(num / den) ** 2)
        out[m / N] = prob
    return out


def _outcome_label(m: int, num_eval_qubits: int) -> str:
    """Tick label for outcome ``m``: its bitstring over the fraction it denotes."""
    from fractions import Fraction

    frac = Fraction(m, 2**num_eval_qubits)
    text = "0" if frac == 0 else (f"{frac.numerator}/{frac.denominator}")
    return f"{m:0{num_eval_qubits}b}\n{text}"


def plot_stage_comparison(
    series: list[tuple[str, dict[str, int], str]],
    num_eval_qubits: int,
    true_phase: float | None = None,
    ax=None,
    title: str | None = None,
    show_theory: bool = False,
):
    """Grouped bar chart of several runs of the same QPE circuit.

    Used to put an ideal simulator, a device noise model and real hardware side by side.

    **The x axis is categorical, deliberately.**  ``n`` estimation qubits can only ever
    return one of ``2**n`` outcomes, so plotting them at their numeric positions on a
    continuous axis invites two misreadings: matplotlib labels round numbers like 0.2 and
    0.6 that the circuit *cannot* return, and a vertical line at the true phase appears to
    lie nearer some bars than others, as though a run whose bar sits closer to the line
    were the more accurate one.  It is not: every run either returns the right outcome or
    does not.  Evenly spaced categories, labelled with the bitstring and the phase it
    denotes, remove both traps.

    Parameters
    ----------
    series
        ``(label, counts, colour)`` per run, drawn left to right within each group.
    true_phase
        Marked if given.  When it is exactly representable the winning outcome is shaded
        rather than crossed by a line, since a line there is redundant with the bars.
        When it falls between two outcomes, as any non-dyadic phase does, it is drawn as
        a line at its true position between the groups, which is the honest picture: no
        outcome is correct, and the distribution straddles the two nearest.
    show_theory
        Overlay the analytic distribution of :func:`expected_phase_distribution`, the
        noiseless prediction the ideal run should reproduce.
    """
    import matplotlib.pyplot as plt

    n = num_eval_qubits
    num_outcomes = 2**n

    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 4.2))

    slots = np.arange(num_outcomes)
    width = 0.8 / max(len(series), 1)

    for i, (label, counts, colour) in enumerate(series):
        dist = counts_to_phases(counts, n)
        heights = [dist.get(m / num_outcomes, 0.0) for m in range(num_outcomes)]
        offsets = slots + (i - (len(series) - 1) / 2) * width
        ax.bar(offsets, heights, width=width, label=label, color=colour, zorder=3)

    if show_theory and true_phase is not None:
        # One target tick per outcome, not a connecting line: the axis is categorical, so
        # a line between outcomes would draw values at x positions that mean nothing.
        theory = expected_phase_distribution(true_phase, n)
        heights = [theory[m / num_outcomes] for m in range(num_outcomes)]
        ax.hlines(
            heights,
            slots - 0.44,
            slots + 0.44,
            color="#222222",
            linestyle=(0, (4, 2)),
            linewidth=1.4,
            label="theory (noiseless)",
            zorder=5,
        )

    if true_phase is not None:
        position = (true_phase % 1.0) * num_outcomes
        nearest = round(position)
        if abs(position - nearest) < 1e-9:
            # Exactly representable: shade the outcome that *is* the answer.
            ax.axvspan(nearest - 0.5, nearest + 0.5, color="#000000", alpha=0.055, zorder=0)
            ax.text(
                nearest,
                1.045,
                f"true θ = {true_phase:g}\nexactly representable",
                ha="center",
                va="top",
                fontsize=8,
                color="#333333",
            )
        else:
            ax.axvline(position, color="k", linestyle=":", linewidth=1.4, zorder=4)
            ax.text(
                position,
                1.045,
                f" true θ = {true_phase:g}\n falls between outcomes",
                ha="left" if position < num_outcomes - 1.5 else "right",
                va="top",
                fontsize=8,
                color="#333333",
            )

    ax.set_xticks(slots)
    ax.set_xticklabels([_outcome_label(m, n) for m in range(num_outcomes)], fontsize=8)
    ax.set_xlim(-0.6, num_outcomes - 0.4)
    ax.set_xlabel(f"measured outcome: bitstring, and the phase θ it encodes (all {num_outcomes})")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1.10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title or f"QPE with {n} estimation qubits")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    return ax


def plot_phase_distribution(
    estimate: PhaseEstimate,
    true_phase: float | None = None,
    ax=None,
    title: str | None = None,
):
    """Bar plot of a measured phase distribution, optionally against theory."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.5))

    phases = sorted(estimate.distribution)
    probs = [estimate.distribution[p] for p in phases]
    width = 0.6 * estimate.resolution

    ax.bar(phases, probs, width=width, label="measured", color="#4C72B0")

    if true_phase is not None:
        theory = expected_phase_distribution(true_phase, estimate.num_eval_qubits)
        ax.plot(
            sorted(theory),
            [theory[p] for p in sorted(theory)],
            "o--",
            color="#C44E52",
            markersize=4,
            linewidth=1,
            label="theory",
        )
        ax.axvline(
            true_phase, color="k", linestyle=":", linewidth=1, label=f"true θ={true_phase:g}"
        )

    ax.set_xlabel("phase θ")
    ax.set_ylabel("probability")
    ax.set_title(title or f"QPE with {estimate.num_eval_qubits} estimation qubits")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    return ax
