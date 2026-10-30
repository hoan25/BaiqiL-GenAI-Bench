"""Prevent a requested GPU smoke from silently passing on CPU."""

import pytest

from scripts.smoke_clip import select_device


@pytest.mark.parametrize(
    "requested,available,expected",
    [("auto", False, "cpu"), ("auto", True, "cuda"), ("cpu", True, "cpu"), ("cuda", True, "cuda")],
)
def test_device_selection(requested, available, expected):
    assert select_device(requested, available) == expected


def test_explicit_cuda_must_not_silently_fall_back():
    with pytest.raises(ValueError, match="unavailable"):
        select_device("cuda", False)


def test_unknown_device_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        select_device("gpu", True)
