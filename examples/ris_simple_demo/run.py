"""CLI entry point mirroring how external researchers would run the demo."""

from __future__ import annotations

import argparse

from acpt.examples.ris_simple_demo import run_simple_ris_episode, visualize_ris_state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the standalone RIS environment demo.")
    parser.add_argument("--steps", type=int, default=5, help="Number of env steps to simulate (default: 5)")
    parser.add_argument(
        "--plot",
        choices=["text", "scatter", "3d"],
        default="text",
        help="Visualization mode. 'scatter' and '3d' require matplotlib.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip displaying matplotlib windows (useful for automated runs).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    history = run_simple_ris_episode(steps=args.steps)
    final = history[-1]
    result = visualize_ris_state(
        final["state"],
        mode=args.plot,
        show_plot=not args.no_show,
        title="Simple RIS Demo Snapshot",
    )
    if args.no_show and args.plot in {"scatter", "3d"} and result is not None:
        import matplotlib.pyplot as plt  # local import to keep optional dependency scoped

        fig, _ = result
        plt.close(fig)
    print(f"Completed {len(history) - 1} steps | Final reward: {final['reward']:.3f}")


if __name__ == "__main__":
    main()
