"""Lightweight registries for the unified WSVAD framework.

Every paper in ``WSAD_INTEGRATION_PLAN.md`` decomposes into the same slots
(temporal encoder, scoring head, loss, optional text/audio branch). Each slot
implementation registers itself with a string name so that Hydra configs can
reference it by name instead of importing a concrete class.

Usage
-----
>>> from src.registry import MODELS
>>> @MODELS.register("rtfm")
... class RTFMForVideoAnomalyDetection(...):
...     ...
>>> cls = MODELS.get("rtfm")
"""

from typing import Any, Callable, Dict, Iterator, List, Optional


class Registry:
    """A name -> object map with a decorator-based registration API."""

    def __init__(self, name: str):
        self._name = name
        self._obj_map: Dict[str, Any] = {}

    def register(
        self, name: Optional[str] = None, obj: Optional[Any] = None
    ) -> Callable:
        """Register ``obj`` under ``name``.

        Can be used as a decorator (``@reg.register("foo")``) or called
        directly (``reg.register("foo", obj)``).
        """

        def _do_register(target: Any, key: Optional[str]) -> Any:
            key = key or getattr(target, "__name__", None)
            if key is None:
                raise ValueError("A registration name is required.")
            if key in self._obj_map:
                raise KeyError(
                    f"'{key}' is already registered in '{self._name}' registry."
                )
            self._obj_map[key] = target
            return target

        # direct call form
        if obj is not None:
            return _do_register(obj, name)

        # decorator form
        def deco(target: Any) -> Any:
            return _do_register(target, name)

        return deco

    def get(self, name: str) -> Any:
        if name not in self._obj_map:
            raise KeyError(
                f"'{name}' not found in '{self._name}' registry. "
                f"Available: {sorted(self._obj_map)}"
            )
        return self._obj_map[name]

    def keys(self) -> List[str]:
        return sorted(self._obj_map)

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __iter__(self) -> Iterator[str]:
        return iter(self._obj_map)

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, items={self.keys()})"


# Slot registries (see WSAD_INTEGRATION_PLAN.md section 4).
MODELS = Registry("models")  # full assembled anomaly-detection models
ENCODERS = Registry("encoders")  # slot 2: temporal encoders
HEADS = Registry("heads")  # slot 4: MIL / scoring heads
LOSSES = Registry("losses")  # slot 5: loss terms
FEATURE_EXTRACTORS = Registry("feature_extractors")  # slot 1: offline backbones
