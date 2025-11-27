"""Common visualization utilities for ACP scenarios."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping, MutableSequence, Sequence

try:  # Optional dependency for plotting
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - dependency optional at runtime
    plt = None  # type: ignore


def _format_position(position: Sequence[float]) -> str:
    return f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"


def _render_text(state: Mapping[str, Any], *, title: str) -> None:
    users = state.get("users", [])
    snr_values = [user.get("effective_snr_db", 0.0) for user in users]
    avg_snr = fmean(snr_values) if snr_values else 0.0

    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)
    print(f"RIS tiles: {len(state.get('phase_profile', []))} | avg SNR: {avg_snr:.2f} dB")
    print("-" * 64)
    print(f"{'UE':<10}{'Position':<26}{'LoS SNR (dB)':>14}{'Effective (dB)':>14}")
    print("-" * 64)
    for user in users:
        print(
            f"{user.get('id', '?'):<10}"
            f"{_format_position(user.get('pos', (0.0, 0.0, 0.0))):<26}"
            f"{user.get('los_snr_db', 0.0):>14.2f}"
            f"{user.get('effective_snr_db', 0.0):>14.2f}"
        )
    print("-" * 64 + "\n")


def _require_matplotlib() -> Any:
    if plt is None:  # pragma: no cover - executed only when matplotlib missing
        raise ImportError(
            "matplotlib is required for scatter or 3D visualization. Install it via 'pip install matplotlib'."
        )
    return plt


def _extract_xyz(state: Mapping[str, Any]) -> tuple[list[float], list[float], list[float], list[float]]:
    xs: MutableSequence[float] = []
    ys: MutableSequence[float] = []
    zs: MutableSequence[float] = []
    snrs: MutableSequence[float] = []
    for user in state.get("users", []):
        pos = user.get("pos", (0.0, 0.0, 0.0))
        xs.append(float(pos[0]))
        ys.append(float(pos[1]))
        zs.append(float(pos[2]))
        snrs.append(float(user.get("effective_snr_db", 0.0)))
    return list(xs), list(ys), list(zs), list(snrs)


def visualize_ris_state(
    state: Mapping[str, Any],
    *,
    mode: str = "text",
    show_plot: bool = True,
    title: str | None = None,
):
    """Visualize a RIS agent state using text, scatter, or 3D plots.

    Args:
        state: Mapping returned from a RIS environment step/snapshot.
        mode: One of ``"text"``, ``"scatter"``, or ``"3d"``.
        show_plot: When ``True`` and plotting is requested, display the figure.
        title: Optional label used for textual and plotted headers.

    Returns:
        ``None`` for text mode. For plotting modes, returns the ``(figure, axis)`` tuple
        when ``show_plot`` is ``False`` so callers can perform custom rendering.
    """

    mode_key = mode.lower()
    title = title or "RIS Scenario"

    if mode_key == "text":
        _render_text(state, title=title)
        return None

    if mode_key not in {"scatter", "3d"}:
        raise ValueError("mode must be 'text', 'scatter', or '3d'")

    mpl = _require_matplotlib()
    xs, ys, zs, snrs = _extract_xyz(state)
    if not xs:
        raise ValueError("State does not contain any user entries to visualize")

    fig = mpl.figure(figsize=(6, 4))
    if mode_key == "scatter":
        ax = fig.add_subplot(111)
        scatter = ax.scatter(xs, ys, c=snrs, cmap="viridis", s=80)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(title)
    else:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(xs, ys, zs, c=snrs, cmap="viridis", s=80)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)

    fig.colorbar(scatter, ax=ax, label="Effective SNR (dB)")
    fig.tight_layout()

    if show_plot:
        mpl.show()
        return None

    return fig, ax
