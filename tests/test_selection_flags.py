"""selection_flags: white-noise and ECORR selections on any per-TOA flag."""

from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals as s


DATA = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


def test_selection_flags_labels_carry_the_flag_name(psr):
    labels = s.selection_flags('be')(psr)
    raw = np.asarray(psr.flags['be']).astype(str)

    assert set(labels.tolist()) == {'be' + v for v in set(raw.tolist())}


def test_selection_flags_combines_several_flags(psr):
    labels = s.selection_flags(['be', 'fe'])(psr)

    assert all('_' in v for v in set(labels.tolist()))
    assert len(set(labels.tolist())) >= len(set(np.asarray(psr.flags['be']).tolist()))


def test_selection_flags_raises_on_an_absent_flag(psr):
    with pytest.raises(KeyError, match='no flag'):
        s.selection_flags('not_a_flag')(psr)


def test_selection_flags_raises_on_empty_values(psr):
    """An empty flag value raises."""
    with pytest.raises(ValueError, match='empty'):
        s.selection_flags('simul')(psr)


def test_selection_flags_splits_the_white_noise(psr):
    noise = s.makenoise_measurement(psr, tnequad=True,
                                    selection=s.selection_flags('be'))
    pars = noise.getN.params
    labels = {'be' + v for v in set(np.asarray(psr.flags['be']).astype(str).tolist())}

    assert len(pars) == 2 * len(labels)
    assert all(any(lab in p for lab in labels) for p in pars)


def test_selection_flags_warns_about_thin_groups(psr):
    with pytest.warns(UserWarning, match='fewer than'):
        s.selection_flags('be', warn_below=len(psr.toas) + 1)(psr)


def test_selection_flags_on_the_backend_flag_matches_the_backend_selection(psr):
    """On a flag equal to the backend flag, the labels are the backend labels with the flag
    name prepended."""
    lab = s.selection_flags('f')(psr)
    be = np.asarray(s.selection_backend_flags(psr)).astype(str)
    assert np.array_equal(lab, np.char.add('f', be))
