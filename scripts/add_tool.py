"""CLI for registering new tools with the ACP wiring map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from acpt.utils import (
    ManifestValidationError,
    RegistrarError,
    derive_tool_class_name,
    load_manifest,
    register_tool,
    scaffold_tool_file,
    validate_manifest,
)

_DEFAULT_WIRING = Path(__file__).resolve().parents[1] / "acpt" / "config" / "wiring.yaml"
_DEFAULT_TOOLS_DIR = Path(__file__).resolve().parents[1] / "acpt" / "tools" / "custom"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register a tool manifest for ACP orchestration.")
    parser.add_argument("--manifest", required=True, help="Path to the tool manifest JSON file.")
    parser.add_argument(
        "--wiring",
        default=str(_DEFAULT_WIRING),
        help="Path to the wiring.yaml file (defaults to acpt/config/wiring.yaml).",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Optional override for the manifest JSON schema path.",
    )
    parser.add_argument("--module", help="Python module path for the tool implementation.")
    parser.add_argument("--class-name", help="Class name implementing the tool.")
    parser.add_argument(
        "--tools-dir",
        default=str(_DEFAULT_TOOLS_DIR),
        help="Directory where scaffolded tools should be written.",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Generate a skeleton tool file derived from the manifest id.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing wiring entries if the id already exists.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        validated = validate_manifest(manifest, schema_path=args.schema)
        tool_id = validated["id"]

        module_path = args.module
        class_name = args.class_name

        if args.scaffold:
            tools_dir = Path(args.tools_dir)
            class_name = class_name or derive_tool_class_name(tool_id)
            scaffold_path = scaffold_tool_file(tools_dir, tool_id, class_name)
            module_basename = scaffold_path.stem
            module_path = module_path or f"acpt.tools.custom.{module_basename}"
        elif module_path is None:
            parser.error("--module is required when --scaffold is not specified.")

        class_name = class_name or derive_tool_class_name(tool_id)

        register_tool(
            validated,
            Path(args.wiring),
            module=module_path,
            class_name=class_name,
            overwrite=args.force,
        )
        print(f"Registered tool '{tool_id}' with module '{module_path}' and class '{class_name}'.")
        return 0
    except (ManifestValidationError, RegistrarError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
