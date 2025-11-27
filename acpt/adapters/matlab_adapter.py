"""Runtime adapter for interacting with the MATLAB Engine API."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Iterable, Optional

from acpt.utils import get_logger

try:  # pragma: no cover - optional dependency
    import matlab.engine  # type: ignore

except ImportError:  # pragma: no cover
    matlab = None  # type: ignore
else:  # pragma: no cover
    matlab = matlab.engine


class MatlabAdapter:
    """Adapter encapsulating MATLAB Engine lifecycle and invocation."""

    def __init__(
        self,
        *,
        session: Optional[Any] = None,
        start_engine: bool = True,
        shared: bool = False,
    ) -> None:
        self._logger = get_logger(self.__class__.__name__)
        self._engine = session
        self._shared = shared

        if self._engine is None and start_engine:
            self._engine = self._start_engine(shared=shared)

    @staticmethod
    def available() -> bool:
        """Return True if the MATLAB Engine Python package is available."""

        return matlab is not None

    def call(
        self,
        function_name: str,
        *args: Any,
        nargout: int = 1,
        **kwargs: Any,
    ) -> Any:
        """Invoke a MATLAB function and return the raw result."""

        engine = self._ensure_engine()
        try:
            func = getattr(engine, function_name)
        except AttributeError as exc:
            raise AttributeError(f"MATLAB function '{function_name}' not found") from exc

        self._logger.info("Calling MATLAB function '%s' with nargout=%s", function_name, nargout)
        return func(*args, nargout=nargout, **kwargs)

    def close(self) -> None:
        """Terminate the MATLAB session if the adapter owns it."""

        if self._engine is not None and hasattr(self._engine, "quit"):
            self._logger.info("Closing MATLAB engine session")
            self._engine.quit()
        self._engine = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        with contextlib.suppress(Exception):
            self.close()

    # Internal helpers -------------------------------------------------

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            if matlab is None:
                raise RuntimeError(
                    "MATLAB engine is unavailable. Install the MATLAB Engine API or provide a session."
                )
            self._engine = self._start_engine(shared=self._shared)
        return self._engine

    def _start_engine(self, *, shared: bool) -> Any:
        if matlab is None:  # pragma: no cover - guard
            raise RuntimeError("MATLAB Engine API is not installed in this environment")

        if shared:
            self._logger.info("Connecting to shared MATLAB session")
            try:
                return matlab.connect_matlab()
            except Exception as exc:  # pragma: no cover - connection failures
                raise RuntimeError("Failed to connect to shared MATLAB session") from exc

        self._logger.info("Starting dedicated MATLAB engine session")
        return matlab.start_matlab()  # pragma: no cover - requires MATLAB runtime
