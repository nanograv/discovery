#!/usr/bin/env python3
"""Tests for Pulsar feather I/O"""

import json
from pathlib import Path

import numpy as np
import pyarrow.feather
import pytest

import discovery as ds

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FEATHER_FILES = [
    DATA_DIR / "v1p1_de440_pint_bipm2019-B1855+09.feather",
    DATA_DIR / "v1p1_de440_pint_bipm2019-J0023+0923.feather",
]


class CopyingPulsar(ds.Pulsar):
    """Mimics an enterprise Pulsar: array attributes are properties that return a fresh copy."""

    arrays = ["Mmat", "sunssb", "pos_t", "planetssb"]

    def __init__(self, psr):
        self.__dict__.update(psr.__dict__)
        self._arrays = {array: self.__dict__.pop(array) for array in self.arrays}
        self.counts = dict.fromkeys(self.arrays, 0)

    def _get(self, array):
        self.counts[array] += 1
        return self._arrays[array].copy()

    Mmat = property(lambda self: self._get("Mmat"))
    sunssb = property(lambda self: self._get("sunssb"))
    pos_t = property(lambda self: self._get("pos_t"))
    planetssb = property(lambda self: self._get("planetssb"))


@pytest.mark.unit
class TestPulsarFeather:
    @pytest.mark.parametrize("src", FEATHER_FILES, ids=lambda p: p.stem.split("-", 1)[1])
    def test_save_feather_roundtrip(self, src, tmp_path):
        """read_feather -> save_feather reproduces the original file's columns and metadata."""
        out = tmp_path / src.name
        ds.Pulsar.read_feather(src).save_feather(out)

        orig, new = pyarrow.feather.read_table(src), pyarrow.feather.read_table(out)

        assert new.column_names == orig.column_names
        for name in orig.column_names:
            assert new.schema.field(name).type == orig.schema.field(name).type, name
            # not Table.equals: planetssb contains NaNs, and NaN != NaN there
            np.testing.assert_array_equal(new[name].to_numpy(), orig[name].to_numpy(), err_msg=name)

        assert json.loads(new.schema.metadata[b"json"]) == json.loads(orig.schema.metadata[b"json"])

    def test_save_feather_reads_each_array_once(self, tmp_path):
        """Test that array attributes are fetched once, not once per column slice.

        With enterprise Pulsars every fetch is a fresh copy, which blew up memory on large pulsars.
        This intentionally fails on the old code but passes with the new changes to `save_feather`
        which work around this enterprise behavior.
        """
        psr = CopyingPulsar(ds.Pulsar.read_feather(FEATHER_FILES[0]))
        psr.save_feather(tmp_path / "psr.feather")

        assert psr.counts == dict.fromkeys(CopyingPulsar.arrays, 1)
