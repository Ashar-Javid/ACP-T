"""Adapter for executing ns-3 simulations via command-line entrypoints."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from acpt.utils import get_logger


class Ns3Adapter:
    """Adapter wrapping ns-3 command execution with timeout and logging."""

    def __init__(
        self,
        ns3_root: Optional[str | Path] = None,
        *,
        executable: str = "waf",
        timeout: float = 60.0,
    ) -> None:
        self._logger = get_logger(self.__class__.__name__)
        self._ns3_root = Path(ns3_root) if ns3_root else None
        self._executable = executable
        self._timeout = timeout

    def execute(self, arguments: Iterable[str] | None = None) -> subprocess.CompletedProcess[str]:
        """Execute an ns-3 command and return the completed process object."""

        args = list(arguments or [])
        command = self._build_command(args)
        self._logger.info("Running ns-3 command: %s", " ".join(command))

        try:
            return subprocess.run(
                command,
                cwd=str(self._ns3_root) if self._ns3_root else None,
                check=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.CalledProcessError as exc:
            self._logger.error("ns-3 command failed: %s", exc.stderr)
            raise

    def _build_command(self, args: list[str]) -> list[str]:
        if self._ns3_root is not None:
            executable_path = self._ns3_root / self._executable
        else:
            executable_path = Path(self._executable)

        if executable_path.is_file():
            return [str(executable_path), *args]

        resolved = shutil.which(str(executable_path))
        if resolved is None:
            raise FileNotFoundError(f"Unable to locate ns-3 executable '{self._executable}'")
        return [resolved, *args]
