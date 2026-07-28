"""makegp_standard_normal: a proper unit-normal coefficient GP (vs improper 1e40)."""

import numpy as np

import discovery as ds  # noqa: F401
from discovery import signals as sig


class _Pulsar:
    name = "J0000+0000"


def test_standard_normal_has_identity_coefficient_covariance():
    psr = _Pulsar()
    basis = np.random.default_rng(0).standard_normal((12, 4))
    gp = sig.makegp_standard_normal(psr, basis)
    # Coefficient prior covariance is exactly ones (c ~ Normal(0, I)) -- a proper,
    # finite prior, not the improper 1e40 flat limit.
    np.testing.assert_array_equal(np.asarray(gp.Phi.N), np.ones(4))
    improper = sig.makegp_improper(psr, basis)
    assert np.all(np.asarray(improper.Phi.N) >= 1e39)  # improper stays 1e40


def test_standard_normal_structure():
    psr = _Pulsar()
    basis = np.random.default_rng(1).standard_normal((10, 3))
    gp = sig.makegp_standard_normal(psr, basis, name="wm")
    assert gp.project is False
    assert gp.gpname == "wm"
    (key,) = gp.index
    assert key == f"{psr.name}_wm_coefficients(3)"
    # no column normalization: the basis is stored as passed.
    np.testing.assert_allclose(np.asarray(gp.F), basis)
