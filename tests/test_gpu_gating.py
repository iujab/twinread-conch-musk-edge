"""I6: gpu-marked tests run only with TWINREAD_GPU_TESTS=1 (never by the bench's default command), and reader tests
skip, not fail, when the weights are not in the HF cache (fail only with TWINREAD_REQUIRE_WEIGHTS=1, as on NCSA J1)."""
import pytest

import conftest


class _Item:
    def __init__(self, gpu):
        self.keywords = {"gpu": True} if gpu else {}
        self.marks = []

    def add_marker(self, m):
        self.marks.append(m)


def _collect(monkeypatch, env, cuda):
    monkeypatch.setattr(conftest, "_gpu_ok", lambda: cuda)
    if env is None:
        monkeypatch.delenv(conftest.GPU_ENV, raising=False)
    else:
        monkeypatch.setenv(conftest.GPU_ENV, env)
    items = [_Item(True), _Item(False)]
    conftest.pytest_collection_modifyitems(None, items)
    return items


def test_gpu_tests_skip_without_env_even_with_cuda(monkeypatch):
    gpu, cpu = _collect(monkeypatch, None, cuda=True)
    assert len(gpu.marks) == 1 and "TWINREAD_GPU_TESTS=1" in gpu.marks[0].kwargs["reason"] and not cpu.marks
    gpu, _ = _collect(monkeypatch, "0", cuda=True)
    assert len(gpu.marks) == 1


def test_gpu_tests_run_with_env_and_cuda(monkeypatch):
    gpu, cpu = _collect(monkeypatch, "1", cuda=True)
    assert not gpu.marks and not cpu.marks
    gpu, _ = _collect(monkeypatch, "1", cuda=False)
    assert gpu.marks and gpu.marks[0].kwargs["reason"] == "CUDA unavailable"


def test_require_weights_skips_or_fails(monkeypatch):
    monkeypatch.setattr(conftest, "missing_weights", lambda n: [f"org/{n}/model.bin"])
    req = conftest.require_weights.__wrapped__()
    monkeypatch.delenv(conftest.REQUIRE_WEIGHTS_ENV, raising=False)
    with pytest.raises(pytest.skip.Exception, match="org/conch/model.bin"):
        req("conch")
    monkeypatch.setenv(conftest.REQUIRE_WEIGHTS_ENV, "1")
    with pytest.raises(pytest.fail.Exception):
        req("conch")
    monkeypatch.setattr(conftest, "missing_weights", lambda n: [])
    req("conch", "musk")  # all cached: no skip


def test_missing_weights_detects_uncached_repo(monkeypatch):
    monkeypatch.setitem(conftest.WEIGHT_FILES, "nope", ("twinread-test/not-a-repo-xyz", ["model.bin", ("a", "b")]))
    assert conftest.missing_weights("nope") == ["twinread-test/not-a-repo-xyz/model.bin",
                                                "twinread-test/not-a-repo-xyz/a|b"]
