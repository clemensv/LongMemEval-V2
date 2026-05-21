"""Registration shim — exposes the aimem-backed Memory to the LongMemEval-V2 harness.

This file lives inside the LongMemEval-V2 fork so the harness's
``memory_modules`` package can register ``aimem_lme`` alongside the
built-in backends. The actual implementation lives in the sibling
``aimem-longmemeval`` package at ``C:\\git\\aimem\\longmemeval_prototype``.

Install that package into the LongMemEval-V2 environment:

    uv pip install -e ..\\aimem\\longmemeval_prototype

or with plain pip:

    pip install -e ..\\aimem\\longmemeval_prototype

Then run the harness with ``--memory-config-path`` pointing at a config
that has ``"memory_type": "aimem_lme"``.
"""

from __future__ import annotations

from aimem_lme.memory import AimemLmeMemory  # noqa: F401  # @register_memory side-effect

__all__ = ["AimemLmeMemory"]
