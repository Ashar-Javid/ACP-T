"""Visualization helpers for the simple RIS scenario."""

from __future__ import annotations

from statistics import fmean
from typing import Any, Mapping, Sequence


def _format_pos(position: Sequence[float]) -> str:
    return f"({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})"


def visualize_ris_layout(state: Mapping[str, Any], *, title: str | None = None) -> None:
    """Render a compact textual overview of the RIS layout and metrics."""

    title = title or "Simple RIS Scenario"
    users = state.get("users", [])
    snr_values = [user.get("effective_snr_db", 0.0) for user in users]
    avg_snr = fmean(snr_values) if snr_values else 0.0

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"RIS tiles: {len(state.get('phase_profile', []))} | avg SNR: {avg_snr:.2f} dB")
    print("-" * 60)
    print(f"{'UE':<10}{'Position':<25}{'LoS SNR (dB)':>12}{'Effective (dB)':>12}")
    print("-" * 60)

    for user in users:
        print(
            f"{user.get('id', '?'):<10}"
            f"{_format_pos(user.get('pos', (0.0, 0.0, 0.0))):<25}"
            f"{user.get('los_snr_db', 0.0):>12.2f}"
            f"{user.get('effective_snr_db', 0.0):>12.2f}"
        )

    print("-" * 60 + "\n")
