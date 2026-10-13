"""makegp_fd_powerlaw: the static frequency-dependent delay over frequency bins, its timing-model
projections, its power-law prior, and the bins it is built on."""

import types
import warnings

import numpy as np
import pytest

import discovery as ds
from discovery import signals as s


def _channelised(fd_orders, nchan=32, nepoch=60, seed=0, spread=3e-3):
    """Channelised TOAs across 900-1700 MHz with channel centres equally spaced in ln(nu) and a
    per-TOA scatter of +-spread in ln(nu), and a timing model of an offset, a spin term, a
    constant DM and FD terms of the given orders."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(np.log(900.0), np.log(1700.0), nchan + 1)
    chan = np.tile(np.arange(nchan), nepoch)
    freqs = np.exp(0.5 * (edges[:-1] + edges[1:]))[chan] * np.exp(rng.uniform(-spread, spread, chan.size))
    mjd = np.repeat(np.floor(rng.uniform(58000.0, 60000.0, nepoch)) + rng.uniform(0.0, 1.0, nepoch), nchan)
    t = (mjd - mjd.min()) / (mjd.max() - mjd.min())
    x = np.log(freqs / 1300.0)
    cols = {'Offset': np.ones_like(t), 'F0': t, 'DM': (freqs / 1300.0) ** -2}
    cols.update({f'FD{k}': x ** k for k in fd_orders})
    n, ra, dec = chan.size, 1.2, -0.82
    return types.SimpleNamespace(
        name='FAKE', freqs=freqs, toas=mjd * 86400.0, stoas=mjd * 86400.0, phi=ra, theta=0.5 * np.pi - dec,
        pos=np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]),
        Mmat=np.column_stack(list(cols.values())), fitpars=list(cols), flags={'chan': chan.astype(str)},
        toaerrs=np.full(n, 1e-6), residuals=rng.normal(scale=1e-6, size=n), backend_flags=np.array(['BE'] * n))


def _names(psr):
    return f'{psr.name}_fd_gp_log10_sigma', f'{psr.name}_fd_gp_gamma'


@pytest.fixture
def restore_priors():
    from discovery import prior
    saved = dict(prior.priordict_standard)
    yield prior
    prior.priordict_standard.clear()
    prior.priordict_standard.update(saved)


def _C(gp, p):
    F = np.asarray(gp.F)
    return F @ np.asarray(gp.Phi.getN(p)) @ F.T


def _rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)

# --- the bins ----------------------------------------------------------------------------

def test_flag_bins_are_channel_indicators_at_the_mean_log_frequency():
    psr = _channelised(())
    F, q = s._fd_bins(psr, 'chan')
    chan = psr.flags['chan'].astype(int)
    assert np.array_equal(F, (chan[:, None] == np.arange(32)[None, :]).astype(float))
    assert np.array_equal(q, [np.mean(np.log(psr.freqs[chan == k])) for k in range(32)])


def test_flag_and_integer_bins_agree_on_channels_equal_in_log_frequency():
    """Channel centres equally spaced in ln(nu), scatter below half the spacing: N equal bins in
    ln(nu) between the lowest and highest TOA frequency are the N channels."""
    psr = _channelised((1, 2))
    Ff, qf = s._fd_bins(psr, 'chan')
    Fi, qi = s._fd_bins(psr, 32)
    assert np.array_equal(Ff, Fi) and np.array_equal(qf, qi)
    a, b = s.makegp_fd_powerlaw(psr, bins='chan'), s.makegp_fd_powerlaw(psr, bins=32)
    assert np.array_equal(np.asarray(a.F), np.asarray(b.F))
    p = {f'{psr.name}_fd_gp_log10_sigma': -7.0, f'{psr.name}_fd_gp_gamma': 3.0}
    assert np.array_equal(np.asarray(a.Phi.getN(p)), np.asarray(b.Phi.getN(p)))


def test_integer_bins_drop_empty_bins():
    psr = _channelised(())
    lo = np.log(psr.freqs) < np.log(1100.0)
    psr.freqs = np.where(lo, psr.freqs * 0.8, psr.freqs)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        F, q = s._fd_bins(psr, 64)
    assert F.shape[1] < 64 and np.all(F.sum(axis=0) > 0) and np.all(F.sum(axis=1) == 1.0)


def test_integer_bins_that_split_channels_warn():
    with pytest.warns(UserWarning, match='splitting'):
        s._fd_bins(_channelised(()), 20)


def test_a_missing_flag_raises():
    with pytest.raises(KeyError, match='has no flag'):
        s.makegp_fd_powerlaw(_channelised(()), bins='not_a_flag')


def test_an_empty_flag_value_raises():
    psr = _channelised(())
    psr.flags['chan'] = psr.flags['chan'].copy()
    psr.flags['chan'][5] = ''
    with pytest.raises(ValueError, match='empty'):
        s.makegp_fd_powerlaw(psr)


def test_flag_bins_that_overlap_in_frequency_raise():
    """A channel index reused across two receivers puts each index in both bands."""
    psr = _channelised(())
    psr.flags['chan'] = (psr.flags['chan'].astype(int) % 16).astype(str)
    with pytest.raises(ValueError, match='overlap in frequency.*receiver flag'):
        s.makegp_fd_powerlaw(psr)


@pytest.mark.parametrize("bins", [1, True, 2.5, None])
def test_bins_must_be_a_flag_or_an_integer_of_at_least_two(bins):
    with pytest.raises(ValueError, match='bins must be'):
        s.makegp_fd_powerlaw(_channelised(()), bins=bins)


# --- the projections ---------------------------------------------------------------------

def test_a_jump_between_groups_of_bins_is_removed_through_the_bins():
    """A jump is a step in frequency, which the polynomials do not reproduce but the bins do."""
    psr = _channelised(())
    upper = psr.flags['chan'].astype(int) >= 16
    psr.Mmat = np.column_stack([psr.Mmat, upper.astype(float)])
    psr.fitpars = psr.fitpars + ['JUMP1']
    gp = s.makegp_fd_powerlaw(psr)
    assert gp.node_removed == ['Offset', 'DM', 'JUMP1']
    g = (np.arange(32) >= 16).astype(float)
    assert np.linalg.norm(gp.node_basis.T @ g) < 1e-12 * np.linalg.norm(g)


def test_fewer_than_two_bin_directions_left_raises():
    """Three channels against an offset, a DM and FD1 leave no direction."""
    with pytest.raises(ValueError, match='at least two'):
        s.makegp_fd_powerlaw(_channelised((1,), nchan=3))


def test_the_basis_is_orthogonal_to_the_timing_model():
    psr = _channelised((1, 2))
    gp = s.makegp_fd_powerlaw(psr)
    Q = np.linalg.qr(s.normalise_tm_basis(psr))[0]
    for F in (np.asarray(gp.F), np.asarray(gp.project_basis)):
        assert np.abs(Q.T @ F).max() < 1e-12 * np.abs(F).max()


# --- the prior ---------------------------------------------------------------------------

def test_gamma_zero_is_close_to_independent_offsets_on_uniform_bins():
    """On 32 bins exactly uniform in ln(nu), spacing dq, a flat spectrum from f_lo = 1/(62 dq) to
    the Nyquist frequency 1/(2 dq) correlates bins m apart by -sin(pi m / 31) / (pi m 30 / 31),
    at most 0.034 in magnitude."""
    gp = s.makegp_fd_powerlaw(_channelised((), spread=0.0))
    q = np.log(gp.fd_nodes)
    K = np.asarray(s._fd_powerlaw_correlation(q, *gp.fd_bands[0], 128)(0.0))
    m = np.abs(np.arange(32)[:, None] - np.arange(32)[None, :])
    want = np.where(m == 0, 1.0, -np.sin(np.pi * m / 31) / (np.pi * np.maximum(m, 1) * 30 / 31))
    assert np.allclose(K, want, rtol=0.0, atol=1e-9)
    assert np.abs(K - np.eye(32)).max() < 0.034


def test_the_node_projection_removes_the_frequency_only_columns_exactly():
    """Offset, FDk polynomials and a constant nu^-2 count; anything that varies in time does
    not. The returned basis is orthogonal to each column function at the bin positions."""
    rng = np.random.default_rng(3)
    n = 3000
    freqs = rng.uniform(800.0, 1700.0, n)
    t = rng.uniform(0.0, 1.0, n)
    x = np.log(freqs / 1000.0)
    cols = {'Offset': np.ones(n), 'FD1': x, 'FD2': x ** 2, 'FD5': x ** 5, 'DM': (freqs / 1000.0) ** -2,
            'F0': t, 'DM1': t * (freqs / 1000.0) ** -2, 'PX': np.sin(6.0 * t)}
    fake = types.SimpleNamespace(name='FAKE', freqs=freqs, Mmat=np.column_stack(list(cols.values())),
                                 fitpars=list(cols))
    q = np.log(np.linspace(820.0, 1680.0, 24))
    fmat_raw = np.zeros((n, 24))
    B, labels = s._fd_node_projection(fake, fmat_raw, q)
    assert labels == ['Offset', 'FD1', 'FD2', 'FD5', 'DM']
    assert B.shape == (24, 24 - 5)
    assert np.allclose(B.T @ B, np.eye(B.shape[1]), rtol=0.0, atol=1e-12)
    xq = q - np.log(1000.0)
    for want in (np.ones(24), xq, xq ** 2, xq ** 5, np.exp(-2.0 * xq)):
        assert np.linalg.norm(B.T @ want) < 1e-8 * np.linalg.norm(want)


def test_without_fd_in_the_timing_model_only_the_offset_and_dm_leave_the_prior():
    """With the FD terms dropped from the timing model the process carries the rest of the
    static delay: the polynomials in log-frequency stay in its prior."""
    with_fd = s.makegp_fd_powerlaw(_channelised((1, 2, 3)))
    no_fd = s.makegp_fd_powerlaw(_channelised(()))
    assert with_fd.node_removed == ['Offset', 'DM', 'FD1', 'FD2', 'FD3']
    assert no_fd.node_removed == ['Offset', 'DM']
    assert np.asarray(with_fd.F).shape[1] == 32 - 5 and np.asarray(no_fd.F).shape[1] == 32 - 2

    q = np.log(no_fd.fd_nodes / 1300.0)
    B = no_fd.node_basis
    for g in (np.ones_like(q), np.exp(-2.0 * q)):
        assert np.linalg.norm(B.T @ g) < 1e-12 * np.linalg.norm(g)
    for k in (1, 2, 3):
        assert np.linalg.norm(B.T @ q ** k) > 0.1 * np.linalg.norm(q ** k)


def test_the_kernel_is_the_band_limited_power_law():
    """Exact at gamma = 0, where the step spectrum is the spectrum; within 1e-3 of k(0) at
    steep gamma on the default bin count."""
    from scipy.integrate import quad
    f_lo, f_hi = 0.8, 27.0
    lags = np.concatenate([[0.0], np.linspace(0.005, 0.6, 40)])
    for gamma, tol in ((0.0, 1e-12), (4.0, 1e-3), (8.0, 1e-3)):
        Z = quad(lambda f: f ** -gamma, f_lo, f_hi, limit=400)[0]
        exact = np.array([quad(lambda f: f ** -gamma * np.cos(2 * np.pi * f * L), f_lo, f_hi,
                               limit=400)[0] / Z for L in lags[1:]])
        k = np.asarray(s._fd_powerlaw_correlation(lags, f_lo, f_hi, 128)(gamma))[0, 1:]
        assert np.max(np.abs(k - exact)) < tol


def test_the_band_follows_the_bin_grid():
    gp = s.makegp_fd_powerlaw(_channelised(()), flo_pad=3.0)
    q = np.log(np.asarray(gp.fd_nodes))
    f_lo, f_hi = gp.fd_bands[0]
    assert f_hi == pytest.approx(1.0 / (2.0 * np.median(np.diff(np.sort(q)))), rel=1e-12)
    assert f_lo == pytest.approx(1.0 / (3.0 * (q.max() - q.min())), rel=1e-12)


def test_steeper_spectra_correlate_neighbouring_bins_more():
    gp = s.makegp_fd_powerlaw(_channelised(()))
    q = np.log(gp.fd_nodes)
    corr = s._fd_powerlaw_correlation(q, *gp.fd_bands[0], 128)
    lag1 = []
    for g in (0.0, 2.0, 4.0):
        lag1.append(np.mean(np.diag(np.asarray(corr(g)), 1)))
    assert lag1[0] < lag1[1] < lag1[2]


@pytest.mark.parametrize("bins", ['chan', 20])
def test_sigma_is_the_rms_delay_per_toa_after_projection(bins):
    psr = _channelised((1, 2))
    sig, gam = _names(psr)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gp = s.makegp_fd_powerlaw(psr, bins=bins)
    for g in (0.0, 3.0, 8.0):
        C = _C(gp, {sig: -7.0, gam: g})
        assert np.sqrt(np.mean(np.diag(C))) == pytest.approx(1e-7, rel=1e-10)


def test_the_prior_is_positive_definite_across_its_box():
    sig, gam = _names(_channelised(()))
    for psr in (_channelised(()), _channelised((1, 2, 3, 4)), _channelised((), nchan=64)):
        gp = s.makegp_fd_powerlaw(psr)
        for g in (0.0, 4.0, 8.0):
            w = np.linalg.eigvalsh(np.asarray(gp.Phi.getN({sig: -7.0, gam: g})))
            assert w.min() > 0.0 and w.max() / w.min() < 1e10


def test_the_jitter_floor_scales_with_the_projected_prior():
    psr = _channelised((1, 2, 3), nchan=64)
    """jitter is relative to the projected prior: 1e-8 and 1e-14 give delay covariances
    within 1e-6 of each other at the steepest index."""
    sig, gam = _names(psr)
    p = {sig: -7.0, gam: 8.0}
    a = s.makegp_fd_powerlaw(psr, jitter=1e-8)
    b = s.makegp_fd_powerlaw(psr, jitter=1e-14)
    assert _rel(_C(a, p), _C(b, p)) < 1e-6


def test_gradients_reach_every_hyperparameter():
    psr = _channelised((1,))
    import jax
    sig, gam = _names(psr)
    gp = s.makegp_fd_powerlaw(psr)
    psl = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr), gp])
    p = {k: (1.0 if k.endswith('efac') else -6.5 if 'equad' in k else -7.0) for k in psl.logL.params}
    p[gam] = 3.0
    g = jax.grad(psl.logL)(p)
    for k in (sig, gam):
        assert np.isfinite(float(g[k])) and float(g[k]) != 0.0


def test_an_outer_scale_above_the_bin_nyquist_raises():
    """f_hi = 1/(2 dq) and f_lo = 1/(flo_pad x 31 dq) on 32 equal bins: flo_pad < 2/31 inverts them."""
    with pytest.raises(ValueError, match="Nyquist"):
        s.makegp_fd_powerlaw(_channelised(()), flo_pad=0.05)


def test_project_basis_is_the_span_before_the_node_reduction():
    gp = s.makegp_fd_powerlaw(_channelised((1, 2)))
    P = np.asarray(gp.project_basis)
    assert P.shape[1] == 32 > np.asarray(gp.F).shape[1] == 32 - 4
    assert np.array_equal(np.asarray(gp.F), P @ gp.node_basis)


def test_the_default_priors_cover_the_hyperparameters():
    psr = _channelised((1,))
    import jax
    from discovery import prior
    sig, gam = _names(psr)
    assert prior.getprior_uniform(sig) == [-10, -4] and prior.getprior_uniform(gam) == [0, 7]
    m = ds.PulsarLikelihood([psr.residuals, ds.makenoise_measurement(psr),
                             s.makegp_timing(psr, svd=True), s.makegp_fd_powerlaw(psr)])
    t = prior.makelogtransform_uniform(m.logL)
    y = np.zeros(len(t.params))
    assert np.isfinite(float(t(y))) and np.all(np.isfinite(np.asarray(jax.grad(t)(y))))


def test_signals_registers_no_fd_prior(restore_priors):
    before = dict(restore_priors.priordict_standard)
    s.makegp_fd_powerlaw(_channelised((1,)))
    assert dict(restore_priors.priordict_standard) == before


def test_all_zero_timing_model_columns_are_dropped_with_a_warning():
    psr = _channelised((1,))
    psr.Mmat = np.column_stack([psr.Mmat, np.zeros(len(psr.freqs))])
    psr.fitpars = psr.fitpars + ['JUMP_EMPTY']
    with pytest.warns(UserWarning, match='JUMP_EMPTY'):
        M = s.normalise_tm_basis(psr)
    assert M.shape[1] == psr.Mmat.shape[1] - 1
    assert np.allclose(np.linalg.norm(M, axis=0), 1.0, rtol=1e-14, atol=0)


@pytest.mark.parametrize("kw", [dict(flo_pad=0.0), dict(flo_pad=-2.0), dict(flo_pad=np.inf), dict(nf=0),
                                dict(nf=2.5), dict(jitter=-1e-8), dict(jitter=np.nan)])
def test_out_of_range_settings_raise(kw):
    with pytest.raises(ValueError, match=list(kw)[0]):
        s.makegp_fd_powerlaw(_channelised(()), **kw)


def test_integer_bins_on_continuous_frequencies_do_not_warn():
    psr = _channelised(())
    psr.freqs = np.exp(np.random.default_rng(1).uniform(np.log(900.0), np.log(1700.0), psr.freqs.size))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        s._fd_bins(psr, 16)


def test_a_single_frequency_raises():
    psr = _channelised(())
    psr.freqs = np.full(psr.freqs.size, 1284.0)
    with pytest.raises(ValueError, match='one frequency'):
        s._fd_bins(psr, 'chan')


def test_the_overlap_error_names_the_bins():
    psr = _channelised(())
    psr.flags['chan'] = (psr.flags['chan'].astype(int) % 16).astype(str)
    with pytest.raises(ValueError, match=r"e\.g\. values '\d+' and '\d+'"):
        s._fd_bins(psr, 'chan')


# --- selection: systems that observe the same band ------------------------------------------

def _two_systems(thin=False, seed=2):
    """Two systems over 900-1700 MHz with 32 and 16 channels, each numbering its channels from 0,
    a jump between them in the timing model, and optionally a third system with two channels."""
    parts = [('A', 32), ('B', 16)] + ([('C', 2)] if thin else [])
    psrs = [_channelised((), nchan=n, nepoch=30, seed=seed + i) for i, (_, n) in enumerate(parts)]
    cat = lambda key: np.concatenate([getattr(p, key) for p in psrs])
    sys_ = np.concatenate([[lab] * len(p.freqs) for (lab, _), p in zip(parts, psrs)])
    psr = types.SimpleNamespace(**{k: cat(k) for k in ('freqs', 'toas', 'stoas', 'toaerrs', 'residuals', 'backend_flags')})
    psr.name, psr.phi, psr.theta, psr.pos = 'FAKE', psrs[0].phi, psrs[0].theta, psrs[0].pos
    psr.flags = {'chan': np.concatenate([p.flags['chan'] for p in psrs]), 'sys': sys_}
    t = (psr.toas - psr.toas.min()) / np.ptp(psr.toas)
    psr.Mmat = np.column_stack([np.ones_like(t), t, (psr.freqs / 1300.0) ** -2, (sys_ == 'B').astype(float)])
    psr.fitpars = ['Offset', 'F0', 'DM', 'JUMP_B']
    return psr


def _by_system(psr):
    return psr.flags['sys']


def test_systems_sharing_a_band_need_a_selection():
    psr = _two_systems()
    with pytest.raises(ValueError, match='overlap in frequency.*selection='):
        s.makegp_fd_powerlaw(psr)
    gp = s.makegp_fd_powerlaw(psr, selection=_by_system)
    P = np.asarray(gp.project_basis)
    assert P.shape[1] == 32 + 16
    assert sorted(gp.fd_nodes) == ['A', 'B'] and len(gp.fd_nodes['A']) == 32 and len(gp.fd_nodes['B']) == 16
    assert gp.node_removed == ['Offset', 'DM', 'JUMP_B']
    assert np.asarray(gp.F).shape[1] == 48 - 3


def test_each_system_has_its_own_bins_and_band():
    psr = _two_systems()
    gp = s.makegp_fd_powerlaw(psr, selection=_by_system)
    F_raw = np.vstack([s._fd_bins(psr, 'chan', sel=psr.flags['sys'] == lab)[0].T for lab in ('A', 'B')]).T
    assert np.all(F_raw[psr.flags['sys'] == 'B', :32] == 0) and np.all(F_raw[psr.flags['sys'] == 'A', 32:] == 0)
    for (f_lo, f_hi), lab in zip(gp.fd_bands, ('A', 'B')):
        q = np.log(gp.fd_nodes[lab])
        assert f_hi == pytest.approx(1.0 / (2.0 * np.median(np.diff(np.sort(q)))), rel=1e-12)
        assert f_lo == pytest.approx(1.0 / (2.0 * np.ptp(q)), rel=1e-12)


def test_with_a_selection_sigma_is_still_the_rms_delay_per_toa():
    psr = _two_systems()
    gp = s.makegp_fd_powerlaw(psr, selection=_by_system)
    sig, gam = _names(psr)
    for g in (0.0, 4.0):
        assert np.sqrt(np.mean(np.diag(_C(gp, {sig: -7.0, gam: g})))) == pytest.approx(1e-7, rel=1e-10)


def test_a_system_too_thin_for_the_band_gets_no_bins():
    psr = _two_systems(thin=True)
    with pytest.warns(UserWarning, match="'C' gets no FD bins"):
        gp = s.makegp_fd_powerlaw(psr, selection=_by_system)
    assert sorted(gp.fd_nodes) == ['A', 'B']
    assert np.asarray(gp.project_basis).shape[1] == 32 + 16


def test_overlapping_bins_within_one_system_still_raise():
    psr = _two_systems()
    chan = psr.flags['chan'].astype(int)
    psr.flags['chan'] = np.where(psr.flags['sys'] == 'A', chan % 8, chan).astype(str)
    with pytest.raises(ValueError, match='overlap in frequency'):
        s.makegp_fd_powerlaw(psr, selection=_by_system)
