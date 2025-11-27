"""Adapter for running PyTorch inference within ACP tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from acpt.utils import get_logger

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore


class PytorchAdapter:
    """Adapter encapsulating PyTorch model loading and inference."""

    def __init__(self, model: Optional[Any] = None, *, device: Optional[str] = None) -> None:
        self._logger = get_logger(self.__class__.__name__)
        self._device = device or ("cpu" if torch is None else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model = None
        if model is not None:
            self.load_model(model)

    def load_model(self, model: Any) -> None:
        """Load a torch.nn.Module or a path to a serialized state dict."""

        if torch is None:
            raise RuntimeError("PyTorch is not installed. Install torch to use the adapter.")

        if isinstance(model, nn.Module):
            self._model = model.to(self._device)
            self._model.eval()
        else:
            path = Path(model)
            if not path.exists():
                raise FileNotFoundError(f"Model file not found: {path}")
            self._model = torch.jit.load(str(path)).to(self._device)
            self._model.eval()

    def predict(self, tensor: Iterable[float] | np.ndarray | Any) -> np.ndarray:
        """Run inference and return a NumPy array of predictions."""

        if torch is None or self._model is None:
            self._logger.warning("Falling back to numpy identity prediction (no PyTorch available).")
            array = np.asarray(list(tensor) if not isinstance(tensor, np.ndarray) else tensor)
            return array

        with torch.no_grad():
            input_tensor = torch.as_tensor(tensor, dtype=torch.float32, device=self._device)
            output = self._model(input_tensor)
            if isinstance(output, torch.Tensor):
                return output.detach().cpu().numpy()
            raise TypeError("Model output must be a torch.Tensor")
