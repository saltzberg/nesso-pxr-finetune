"""Synthetic seed-contract tests; no assay fitting or performance evidence."""
import copy
import importlib

import torch
from torch import nn

v2 = importlib.import_module('experiments.20260906_affinity_comparison.code.train_v2')


def models():
    result = nn.ModuleDict()
    for name in ['affinity_module', 'affinity_module2']:
        model = nn.Module()
        model.projection = nn.Linear(3, 3)
        model.head = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 1))
        result[name] = model
    return result


def protocol(seed=42):
    return {'training': {'seed': seed, 'lora_targets': ['projection'], 'rank': 2,
                         'alpha': 2, 'continuous_head': 'head'}}


def adapters(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if 'lora_A' in n}


def test_initialization_independent_of_preceding_packet_rng():
    initial = models()
    a, b = copy.deepcopy(initial), copy.deepcopy(initial)
    torch.manual_seed(100)
    torch.rand(71)
    v2.seeded_configure(a, 'head_plus_lora', protocol())
    torch.manual_seed(999)
    torch.rand(313)
    v2.seeded_configure(b, 'head_plus_lora', protocol())
    aa, bb = adapters(a), adapters(b)
    assert aa and aa.keys() == bb.keys()
    assert all(torch.equal(aa[k], bb[k]) for k in aa)


def test_constructor_preserves_caller_rng_and_seed_is_effective():
    initial = models()
    torch.manual_seed(1717)
    before = torch.get_rng_state().clone()
    a, b = copy.deepcopy(initial), copy.deepcopy(initial)
    v2.seeded_configure(a, 'head_plus_lora', protocol(42))
    assert torch.equal(torch.get_rng_state(), before)
    v2.seeded_configure(b, 'head_plus_lora', protocol(43))
    assert any(not torch.equal(adapters(a)[k], adapters(b)[k]) for k in adapters(a))


def test_both_arms_preserve_the_same_pretrained_heads():
    initial = models()
    a, b = copy.deepcopy(initial), copy.deepcopy(initial)
    v2.seeded_configure(a, 'head_only', protocol())
    v2.seeded_configure(b, 'head_plus_lora', protocol())
    assert all(torch.equal(p, dict(b.named_parameters())[n]) for n,p in a.named_parameters() if '.head.' in n)


def test_wrapper_is_part_of_bound_sources():
    paths = v2.versioned_sources(v2.base.read_json(v2.base.PROTOCOL), True)
    assert 'experiments/20260906_affinity_comparison/code/train_v2.py' in paths
    assert 'tests/test_affinity_comparison_seed_isolation.py' in paths
