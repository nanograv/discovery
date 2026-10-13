"""makegp_pa: the delay locked to twice the parallactic angle per frequency bin, and the angle it
is built on."""

import types
import warnings
from pathlib import Path

import numpy as np
import pytest

import discovery as ds
from discovery import signals as s
from discovery import likelihood as dl


DATA = Path(__file__).resolve().parent.parent / "data"

# site passed to parallactic_angle; any key of OBSERVATORY_ITRF gives a valid geometry
SITE = 'meerkat'


@pytest.fixture(scope="module")
def psr():
    f = DATA / "v1p1_de440_pint_bipm2019-J0030+0451.feather"
    if not f.exists():
        pytest.skip("pulsar data fixture missing")
    return ds.Pulsar.read_feather(f)


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


# --- the angle -------------------------------------------------------------------------

def test_gmst_matches_the_defining_value_at_j2000():
    """GMST at J2000.0 is 18h 41m 50.5482s, i.e. 280.46061837 degrees."""
    got = np.rad2deg(s.greenwich_sidereal_angle(51544.5))
    assert abs(got - 280.46061837) < 1e-6


def test_gmst_advances_by_one_sidereal_day():
    """A mean sidereal day is 86164.0905 s, not 86400."""
    a = s.greenwich_sidereal_angle(58000.0)
    b = s.greenwich_sidereal_angle(58000.0 + 86164.0905 / 86400.0)
    assert abs((np.rad2deg(b - a) + 180.0) % 360.0 - 180.0) < 1e-4


def test_parallactic_angle_vanishes_at_transit(psr):
    """At zero hour angle the source, the zenith and the pole are on one great circle.

    The angle is then 0 or pi, according to whether the source transits south or north
    of the zenith, so it is sin(psi) that must vanish and not psi itself.
    """
    lat, lon = s.observatory_location(SITE)
    # solve for the MJDs at which H = 0 by stepping the sidereal rate
    mjd = 58000.0
    for _ in range(60):
        H = s.greenwich_sidereal_angle(mjd) + lon - psr.phi
        mjd -= float(np.arctan2(np.sin(H), np.cos(H))) / (2 * np.pi) * (86164.0905 / 86400.0)

    class AtTransit:
        name, phi, theta = psr.name, psr.phi, psr.theta
        stoas = np.array([mjd * 86400.0])

    assert abs(float(np.sin(s.parallactic_angle(AtTransit, site=SITE)[0]))) < 1e-5


def test_parallactic_angle_uses_site_arrival_times_not_barycentric(psr):
    """psr.toas carry the Roemer delay, which is degrees of hour angle."""
    psi = s.parallactic_angle(psr, site=SITE)

    class Barycentred:
        name, phi, theta = psr.name, psr.phi, psr.theta
        stoas = psr.toas

    other = s.parallactic_angle(Barycentred, site=SITE)
    shift = np.max(np.abs((np.rad2deg(psi - other) + 180.0) % 360.0 - 180.0))
    assert shift > 0.1, "swapping stoas for toas must change the angle appreciably"


def test_parallactic_angle_agrees_with_astropy(psr):
    """Against an independent apparent-sidereal-time implementation."""
    astropy = pytest.importorskip("astropy")
    from astropy.time import Time
    from astropy.coordinates import EarthLocation
    import astropy.units as u

    lat, lon = s.observatory_location(SITE)
    loc = EarthLocation(lat=np.rad2deg(lat) * u.deg, lon=np.rad2deg(lon) * u.deg)
    mjd = np.asarray(psr.stoas) / 86400.0
    t = Time(mjd, format="mjd", scale="utc", location=loc)
    H = (t.sidereal_time("apparent").radian + 0.0) - psr.phi
    dec = 0.5 * np.pi - psr.theta
    ref = np.arctan2(np.sin(H) * np.cos(lat),
                     np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(H))

    d = (np.rad2deg(s.parallactic_angle(psr, site=SITE) - ref) + 180.0) % 360.0 - 180.0
    assert np.max(np.abs(d)) < 0.05


def test_observatory_lookup_by_code_and_by_explicit_coordinates():
    assert np.allclose(np.rad2deg(s.observatory_location('pks')),
                       [-32.998406, 148.263510], atol=1e-5)
    assert np.allclose(np.rad2deg(s.observatory_location('meerkat')),
                       [-30.711056, 21.443889], atol=1e-5)
    assert np.allclose(np.rad2deg(s.observatory_location((-30.0, 21.0))), [-30.0, 21.0])


def test_unknown_site_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no coordinates"):
        s.observatory_location('not_a_telescope')


# --- the basis ---------------------------------------------------------------------------

def _raw(psr, bins='chan'):
    """[F_FD sin(2 psi) | F_FD cos(2 psi)] from the indicator basis and the angle."""
    ind, _ = s._fd_bins(psr, bins)
    psi = s.parallactic_angle(psr, site=SITE)
    return np.hstack([ind * np.sin(2 * psi)[:, None], ind * np.cos(2 * psi)[:, None]])


def test_the_basis_is_the_projected_sin_and_cos_of_twice_the_angle_per_bin():
    psr = _channelised((1,))
    Q = np.linalg.qr(s.normalise_tm_basis(psr))[0]
    raw = _raw(psr)
    want = raw - Q @ (Q.T @ raw)
    gp = s.makegp_pa(psr, site=SITE)
    assert np.asarray(gp.F).shape == (len(psr.freqs), 64)
    assert np.allclose(np.asarray(gp.F), want, rtol=0.0, atol=1e-13)
    assert np.allclose(gp.pa_bins, np.exp(s._fd_bins(psr, 'chan')[1]), rtol=1e-15, atol=0.0)


def test_one_scale_governs_the_whole_basis():
    psr = _channelised(())
    gp = s.makegp_pa(psr, site=SITE)
    key = f'{psr.name}_pa_gp_log10_sigma'
    assert gp.Phi.params == [key]
    assert np.array_equal(np.asarray(gp.Phi.getN({key: -7.0})), np.full(64, 1e-14))


def test_integer_bins_give_the_flag_bins_on_channels_equal_in_log_frequency():
    psr = _channelised(())
    a, b = s.makegp_pa(psr, site=SITE), s.makegp_pa(psr, bins=32, site=SITE)
    assert np.array_equal(np.asarray(a.F), np.asarray(b.F))


def test_project_removes_the_whole_fd_span():
    psr = _channelised((1, 2))
    fd = s.makegp_fd_powerlaw(psr)
    U, S, _ = np.linalg.svd(np.asarray(fd.project_basis), full_matrices=False)
    Qfd = U[:, S > 1e-10 * S[0]]
    with_fd = np.asarray(s.makegp_pa(psr, site=SITE, project=fd).F)
    without = np.asarray(s.makegp_pa(psr, site=SITE).F)
    assert np.abs(Qfd.T @ with_fd).max() < 1e-10 * np.abs(with_fd).max()
    assert np.abs(Qfd.T @ without).max() > 1e-3 * np.abs(without).max()


@pytest.mark.parametrize("project", [np.zeros((10, 2)), types.SimpleNamespace(F=np.zeros((10, 2)))])
def test_project_takes_only_an_fd_gp(project):
    with pytest.raises(TypeError, match='project must be None or an FD GP'):
        s.makegp_pa(_channelised(()), site=SITE, project=project)


def test_the_rank_and_retained_power_report():
    """The columns are kept whatever the projections annihilate; rank and power are reported."""
    psr = _channelised((1, 2))
    fd = s.makegp_fd_powerlaw(psr)
    for project in (None, fd):
        gp = s.makegp_pa(psr, site=SITE, project=project)
        F = np.asarray(gp.F)
        sv = np.linalg.svd(F, compute_uv=False)
        assert F.shape[1] == 64
        assert gp.pa_rank == int(np.sum(sv > 1e-8 * sv[0]))
        assert gp.pa_retained == pytest.approx(np.sum(F ** 2) / np.sum(_raw(psr) ** 2), rel=1e-12)
        assert 0.0 < gp.pa_retained <= 1.0


def test_a_missing_bin_flag_raises():
    with pytest.raises(KeyError, match='has no flag'):
        s.makegp_pa(_channelised(()), bins='not_a_flag', site=SITE)


def test_the_default_prior_covers_the_scale_and_building_registers_none():
    from discovery import prior
    before = dict(prior.priordict_standard)
    psr = _channelised(())
    s.makegp_pa(psr, site=SITE)
    assert dict(prior.priordict_standard) == before
    assert prior.getprior_uniform(f'{psr.name}_pa_gp_log10_sigma') == [-10, -6]



def _K(gp, params, idx):
    """Implied delay covariance F Phi F^T on a subset of TOAs, for either Phi shape."""
    F = np.asarray(gp.F)[idx]
    Phi = np.asarray(gp.Phi.getN(params))
    return F @ (Phi @ F.T) if Phi.ndim == 2 else F @ (Phi[:, None] * F.T)


def _shifted(gp, chi):
    """The GP with the origin of psi moved by chi: its columns are [sin 2(psi + chi) |
    cos 2(psi + chi)], rotated from the unshifted blocks."""
    F = np.asarray(gp.F)
    n = F.shape[1] // 2
    c, sn = np.cos(2 * chi), np.sin(2 * chi)
    return types.SimpleNamespace(F=np.hstack([F[:, :n] * c + F[:, n:] * sn,
                                              F[:, n:] * c - F[:, :n] * sn]), Phi=gp.Phi)


def test_the_prior_is_invariant_to_the_psi_origin():
    """Shifting the origin of psi leaves the implied delay covariance unchanged."""
    psr = _channelised(())
    gp = s.makegp_pa(psr, site=SITE)
    idx = np.arange(0, len(psr.toas), 23)[:300]
    sx, chi = 3e-8, 0.35
    Ki0 = _K(_shifted(gp, 0.0), {f'{psr.name}_pa_gp_log10_sigma': np.log10(sx)}, idx)
    Kic = _K(_shifted(gp, chi), {f'{psr.name}_pa_gp_log10_sigma': np.log10(sx)}, idx)
    assert np.abs(Kic - Ki0).max() / np.abs(Ki0).max() < 1e-12



def test_injected_pa_delay_is_recovered_at_the_right_scale():
    """Inject a known per-channel sinusoid in PA and scan the marginal likelihood."""
    psr = _channelised(())
    rng = np.random.default_rng(20260827)
    psi = s.parallactic_angle(psr, site=SITE)
    lab = np.asarray(psr.flags['chan']).astype(str)
    order = sorted(set(lab.tolist()), key=int)

    # twice the per-channel amplitude error err / sqrt(n_chan / 2), with err the median
    # TOA error and n_chan the median TOA count per channel
    err = np.asarray(psr.toaerrs)
    per_bin = np.median([int((lab == v).sum()) for v in order])
    sigma_true = float(2.0 * np.median(err) / np.sqrt(0.5 * per_bin))

    delay = np.zeros(len(psi))
    for value in order:
        m = lab == value
        a, b = rng.normal(scale=sigma_true, size=2)
        delay[m] = a * np.sin(2 * psi[m]) + b * np.cos(2 * psi[m])

    res = delay + rng.normal(scale=err)

    pa = s.makegp_pa(psr, site=SITE)
    m = dl.PulsarLikelihood([res, s.makenoise_measurement(psr, {}),
                             s.makegp_timing(psr, svd=True), pa])
    grid = np.linspace(-10.0, -6.0, 41)
    key = f'{psr.name}_pa_gp_log10_sigma'
    base = {k: (1.0 if k.endswith('efac') else -8.5) for k in m.logL.params}
    curve = np.array([float(m.logL({**base, key: g})) for g in grid])

    best = grid[int(np.argmax(curve))]
    assert abs(best - np.log10(sigma_true)) < 0.3, (best, np.log10(sigma_true))
    assert curve.max() - curve[0] > 20.0, "an injected term must beat the no-term end"


def test_project_must_be_the_fd_gp_of_this_pulsar():
    psr, other = _channelised(()), _channelised((), nepoch=50)
    with pytest.raises(ValueError, match='not this pulsar'):
        s.makegp_pa(psr, site=SITE, project=s.makegp_fd_powerlaw(other))


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


def test_a_selection_gives_each_system_its_own_pa_bins():
    psr = _two_systems()
    with pytest.raises(ValueError, match='overlap in frequency'):
        s.makegp_pa(psr, site=SITE)
    fd = s.makegp_fd_powerlaw(psr, selection=_by_system)
    gp = s.makegp_pa(psr, selection=_by_system, site=SITE, project=fd)
    F = np.asarray(gp.F)
    assert F.shape[1] == 2 * (32 + 16) and len(gp.pa_bins) == 48
    U, S, _ = np.linalg.svd(np.asarray(fd.project_basis), full_matrices=False)
    Qfd = U[:, S > 1e-10 * S[0]]
    assert np.abs(Qfd.T @ F).max() < 1e-10 * np.abs(F).max()


def test_the_site_is_required():
    with pytest.raises(TypeError, match='site'):
        s.makegp_pa(_channelised(()))
