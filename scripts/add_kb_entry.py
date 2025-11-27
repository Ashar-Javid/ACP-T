"""CLI for appending knowledge-base documents and wiring descriptors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from acpt.utils import (
    RegistrarError,
    append_kb_entry,
    register_knowledge,
)

_DEFAULT_WIRING = Path(__file__).resolve().parents[1] / "acpt" / "config" / "wiring.yaml"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_KB_DIR = _REPO_ROOT / "acpt" / "data" / "kb"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Append a document to a knowledge base store.")
    parser.add_argument("--id", required=True, help="Knowledge descriptor id (e.g., kb.ris).")
    parser.add_argument(
        "--document",
        action="append",
        default=[],
        help="Inline document text to append. Can be specified multiple times.",
    )
    parser.add_argument(
        "--file",
        action="append",
        default=[],
        help="Path to a file whose contents should be appended to the KB.",
    )
    parser.add_argument(
        "--path",
        help="Target KB JSON file. Defaults to acpt/data/kb/<id>.json.",
    )
    parser.add_argument(
        "--wiring",
        default=str(_DEFAULT_WIRING),
        help="Path to wiring.yaml for registering the KB descriptor.",
    )
    parser.add_argument(
        "--skip-wiring",
        action="store_true",
        help="Skip wiring.yaml updates and only append the KB documents.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite wiring descriptor if the id already exists.",
    )
    return parser


def _collect_documents(files: Iterable[str], inline_docs: Iterable[str]) -> list[str]:
    documents = [doc for doc in inline_docs if doc]
    for file_name in files:
        path = Path(file_name)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")
        documents.append(path.read_text(encoding="utf-8"))
    return documents


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        documents = _collect_documents(args.file, args.document)
        if not documents:
            parser.error("At least one document content must be provided via --document or --file.")

        knowledge_id = args.id
        if args.path:
            kb_path = Path(args.path)
            if not kb_path.is_absolute():
                kb_path = (_REPO_ROOT / kb_path).resolve()
        else:
            kb_path = (_DEFAULT_KB_DIR / f"{knowledge_id.replace('.', '_')}.json").resolve()

        for doc in documents:
            append_kb_entry(kb_path, doc)

        if not args.skip_wiring:
            wiring_path = Path(args.wiring)
            wiring_parent = wiring_path.parent.resolve()
            descriptor_path = kb_path
            try:
                relative = descriptor_path.relative_to(wiring_parent)
                path_hint = str(relative).replace("\\", "/")
            except ValueError:
                path_hint = str(descriptor_path)
            register_knowledge(
                knowledge_id,
                {"path": path_hint, "index_type": "file", "prefix": path_hint},
                wiring_path,
                overwrite=args.force,
            )

        print(f"Appended {len(documents)} document(s) to {kb_path}.")
        return 0
    except (RegistrarError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
