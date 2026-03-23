"""Abstract base class for all vision sub-processors.

Every sub-module in the pipeline (face detection, gaze estimation, emotion
classification, etc.) inherits from ``BaseProcessor`` so the pipeline
orchestrator can manage model lifecycle uniformly via ``load()`` / ``unload()``
and the context-manager protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseProcessor(ABC):
    """Abstract base for vision sub-processors.

    Parameters
    ----------
    device : str
        ``"auto"`` (default) selects CUDA when available, otherwise CPU.
        Explicit ``"cpu"`` or ``"cuda"`` values are respected as-is.
    """

    def __init__(self, device: str = "auto") -> None:
        if device == "auto":
            self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Load model weights and allocate resources.

        Concrete implementations **must** call ``super().load()`` at the end
        so that ``is_loaded`` reflects the correct state.
        """
        self._loaded = True

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free GPU/CPU memory.

        Concrete implementations **must** call ``super().unload()`` at the end.
        """
        self._loaded = False

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """Run inference on a single input (typically one frame or face crop).

        The exact signature varies per sub-processor; the ABC only enforces
        that the method exists.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Whether the processor's model/resources are currently loaded."""
        return self._loaded

    def __enter__(self) -> "BaseProcessor":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()
