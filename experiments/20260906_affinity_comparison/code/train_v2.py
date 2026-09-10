"""Seed-isolated initialization over the immutable first trainer implementation.

Native parity evaluation restores packet RNG streams. Isolate adapter construction
so evaluation order cannot replace the declared initialization seed. No old source
or completed smoke artifact is modified; use a distinct run binding/output root.
"""
from __future__ import annotations
import importlib
from pathlib import Path

base = importlib.import_module('experiments.20260906_affinity_comparison.code.train')
original_configure = base.configure
original_sources = base.sources


def seeded_configure(models, arm, protocol):
    import torch
    devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(protocol['training']['seed'])
        return original_configure(models, arm, protocol)


def versioned_sources(protocol, smoke):
    return original_sources(protocol, smoke) + [
        str(Path(__file__).resolve().relative_to(base.ROOT)),
        'tests/test_affinity_comparison_seed_isolation.py',
    ]


def main():
    base.configure = seeded_configure
    base.sources = versioned_sources
    return base.main()

if __name__ == '__main__':
    raise SystemExit(main())
