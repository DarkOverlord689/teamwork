"""Abstract base class for all audio sub-processors.

Every sub-module in the audio pipeline (diarisation, transcription, turn
analysis, participation metrics, etc.) inherits from ``AudioBaseProcessor``
so the pipeline orchestrator can manage model lifecycle uniformly via
``load()`` / ``unload()`` and the context-manager protocol.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AudioBaseProcessor(ABC):
    """Abstract base for audio sub-processors.

    Parameters
    ----------
    device : str
        ``"auto"`` (default) selects CUDA when available, otherwise CPU.
        Explicit ``"cpu"`` or ``"cuda"`` values are respected as-is.
    """

    processor_name: str = "base_audio_processor"

    def __init__(self, device: str = "auto") -> None:
        try:
            import torch  # type: ignore

            if device == "auto":
                self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
        except ImportError:
            # torch not available — fall back to CPU
            self.device = "cpu" if device == "auto" else device
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
        """Run inference on a single input (typically an audio array or path).

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

    def __enter__(self) -> "AudioBaseProcessor":
        self.load()
        return self

    def __exit__(self, *args: Any) -> None:
        self.unload()
