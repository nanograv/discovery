"""Tests for the time-domain covariance kernels and their use in the
solar-wind time-domain GP (makegp_timedomain_solar_dm)."""

from pathlib import Path

import numpy as np
import pytest

from discovery import signals, solar
import discovery as ds


DATA = Path(__file__).resolve().parent.parent / "data"


def _tau():
    t = np.linspace(0.0, 5 * 365.25 * 86400.0, 40)
    return np.abs(np.subtract.outer(t, t))


def test_squared_exponential_shape_symmetry_and_diag():
    tau = _tau()
    k = np.asarray(signals.squared_exponential(tau, log10_sigma=-6.0, log10_ell=2.0))
    assert k.shape == tau.shape
    assert np.all(np.isfinite(k))
    assert np.allclose(k, k.T)
    assert np.all(np.diag(k) > 0.0)
    # off-diagonal correlation decreases with separation
    row = k[0]
    assert row[1] >= row[-1]


def test_quasi_periodic_reduces_to_se_as_gamma_zero():
    tau = _tau()
    se = np.asarray(signals.squared_exponential(tau, -6.0, 2.0))
    qp = np.asarray(signals.quasi_periodic(tau, -6.0, 2.0, log10_Gamma=-30.0, log10_p=0.5))
    assert np.allclose(se, qp, rtol=1e-6, atol=1e-30)


@pytest.mark.integration
def test_solar_timedomain_gp_builds_and_samples():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    psr = ds.Pulsar.read_feather(f)
    gp = solar.makegp_timedomain_solar_dm(psr, covariance=signals.squared_exponential,
                                          dt=14 * 86400.0, name="sw_gp")
    model = ds.PulsarLikelihood([psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True),
                                 gp])
    params = model.logL.params
    assert any("sw_gp_log10_sigma" in p for p in params)
    assert any("sw_gp_log10_ell" in p for p in params)
    # priors now live in core prior.py, so sample_uniform resolves them
    p0 = ds.sample_uniform(params)
    assert np.isfinite(float(model.logL(p0)))
