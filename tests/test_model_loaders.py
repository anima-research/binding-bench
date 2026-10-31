"""Loader plugins for non-hf model specs: register_loader, $BINDING_BENCH_LOADERS and entry points, and the refusals."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from binding_bench.binding import models as m

SPEC = m.ModelSpec.parse("koto:/c/step.pt?size=1b-tsumugi")


def _fake(spec: m.ModelSpec, device: str) -> m.LoadedModel:
    return m.LoadedModel(spec, lambda ids: torch.zeros(ids.shape + (4,)), object(), torch.device("cpu"), 128)


def _other(spec: m.ModelSpec, device: str) -> m.LoadedModel:
    return _fake(spec, device)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.setattr(m, "_REGISTERED", {})
    monkeypatch.delenv(m.LOADERS_ENV, raising=False)
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [])


@pytest.fixture
def plugin_module(monkeypatch):
    mod = types.ModuleType("fake_koto_plugin")
    mod.load = _fake
    mod.other = _other
    mod.not_callable = 3
    monkeypatch.setitem(sys.modules, "fake_koto_plugin", mod)
    return mod


def test_no_loader_is_refused_with_every_way_to_provide_one():
    with pytest.raises(m.LoaderMissing, match="register_loader.*BINDING_BENCH_LOADERS.*binding_bench.loaders"):
        m.load(SPEC, device="cuda")


def test_registered_loader_loads():
    m.register_loader("koto", _fake)
    m.register_loader("koto", _fake)   # the same callable again is a no-op
    out = m.load(SPEC, device="cuda")
    assert out.spec == SPEC and out.max_seq_len == 128


def test_a_second_registration_is_refused():
    m.register_loader("koto", _fake)
    with pytest.raises(m.LoaderMissing, match="already registered"):
        m.register_loader("koto", _other)
    with pytest.raises(m.LoaderMissing, match="built in"):
        m.register_loader("hf", _fake)


def test_env_loader_loads(monkeypatch, plugin_module):
    monkeypatch.setenv(m.LOADERS_ENV, "koto=fake_koto_plugin:load")
    assert m.loader_for("koto") is _fake


def test_env_and_registry_agreeing_is_fine_and_disagreeing_is_refused(monkeypatch, plugin_module):
    m.register_loader("koto", _fake)
    monkeypatch.setenv(m.LOADERS_ENV, "koto=fake_koto_plugin:load")
    assert m.loader_for("koto") is _fake
    monkeypatch.setenv(m.LOADERS_ENV, "koto=fake_koto_plugin:other")
    with pytest.raises(m.LoaderMissing, match="different loaders"):
        m.loader_for("koto")


@pytest.mark.parametrize("value, match", [
    ("koto", "kind=module:attr"),
    ("koto=fake_koto_plugin", "module:attr"),
    ("koto=fake_koto_plugin:not_callable", "not callable"),
    ("koto=fake_koto_plugin:load,koto=fake_koto_plugin:other", "two loaders"),
])
def test_bad_env_entries_are_refused(monkeypatch, plugin_module, value, match):
    monkeypatch.setenv(m.LOADERS_ENV, value)
    with pytest.raises(m.LoaderMissing, match=match):
        m.loader_for("koto")


def test_entry_point_loader_loads(monkeypatch):
    class EP:
        name, value = "koto", "pkg.mod:load"

        def load(self):
            return _fake

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [EP()] if group == m.LOADER_GROUP else [])
    assert m.loader_for("koto") is _fake


def test_a_loader_that_returns_the_wrong_type_is_refused():
    m.register_loader("koto", lambda spec, device: "not a model")
    with pytest.raises(m.LoaderMissing, match="not a LoadedModel"):
        m.load(SPEC, device="cuda")


def test_cap_cuda_memory_without_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(m.VRAM_BUDGET_ENV, raising=False)
    assert m.cap_cuda_memory() is None


def test_cap_cuda_memory_rejects_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(m.VRAM_BUDGET_ENV, "lots")
    with pytest.raises(ValueError):
        m.cap_cuda_memory()


def test_cap_cuda_memory_sets_fraction(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, int]] = []

    class Props:
        total_memory = 100 * 1024 ** 3

    monkeypatch.setenv(m.VRAM_BUDGET_ENV, "25")
    monkeypatch.setattr(m.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(m.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(m.torch.cuda, "get_device_properties", lambda i: Props())
    monkeypatch.setattr(m.torch.cuda, "set_per_process_memory_fraction", lambda f, i: calls.append((f, i)))
    assert m.cap_cuda_memory() == pytest.approx(0.25)
    assert calls == [(pytest.approx(0.25), 0), (pytest.approx(0.25), 1)]
