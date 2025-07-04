import numpy as np

from .. import matrix
from .. import signals
from .. import prior
from .. import solar
from .. import likelihood

prior.priordict_standard.update({
    "(.*_)?efac":               [0.5, 2],
    "(.*_)?tnequad":            [-10, -5],
    "(.*_)?log10_ecorr":        [-10, -5],
    '(.*_)?red_noise_log10_A':  [-18, -11],
    '(.*_)?red_noise_gamma':    [0, 7],
    '(.*_)?dm_gp_log10_A':      [-18, -11],
    '(.*_)?dm_gp_gamma':        [0, 7],
    '(.*_)?chrom_gp_log10_A':   [-18, -11],
    '(.*_)?chrom_gp_gamma':     [0, 7],
    '(.*_)?chrom_gp_alpha':     [3, 14],
    '(.*_)?sw_gp_log10_A':      [-10, -2],
    '(.*_)?sw_gp_gamma':        [0, 4],
    '(.*_)?band_gp_log10_A':    [-18, -11],
    '(.*_)?band_gp_gamma':      [0, 7],
    '(.*_)?band_gp_alpha':      [3, 14],
    '(.*_)?band_gp_fcutoff':    [500, 2000],
    'curn_log10_A':             [-18, -11],
    'curn_gamma':               [0, 7],
})

def gps2commongp(gps):
    priors = [gp.Phi.getN for gp in gps]
    pmax = len(gps)
    ns = [gp.F.shape[1] for gp in gps]  # Does not work for callable gp.F (e.g. chromatic GP)
    nmax = max(ns)

    def prior(params):
        yp = matrix.jnp.full((pmax, nmax), 1e-40)
        for i,p in enumerate(priors):
            yp = yp.at[i, :ns[i]].set(p(params))

        return yp

    prior.params = sorted(set([par for p in priors for par in p.params]))
    Fs = [np.pad(gp.F, [(0,0), (0,nmax - gp.F.shape[1])]) for gp in gps]

    return matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(prior), Fs)


def make_psr_gps_fourier(psr, max_cadence_days=14, red=True, dm=True, chrom=True, sw=True, band=True, band_alpha=False):
    psr_Tspan = signals.getspan(psr)
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))

    return (([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, name='red_noise')] if red else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_dm, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp')] if chrom else [])+ \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=solar.fourierbasis_solar_dm, name='sw_gp')] if sw else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_low, name='band_gp')] if band else []) + \
            ([signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_low_alpha, name='band_gp_alpha')] if band_alpha else []))


def make_psr_gps_fftint(psr, max_cadence_days=14, red=True, dm=True, chrom=True, sw=True, band=True, band_alpha=False):
    psr_Tspan = signals.getspan(psr)
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1

    return (([signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, name='red_noise')] if red else []) + \
            ([signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, name='dm_gp')] if dm else [])+ \
            ([signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, name='chrom_gp')] if chrom else [])+ \
            ([signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, name='sw_gp')] if sw else []) + \
            ([signals.makegp_fftcov_band(psr, signals.powerlaw, components=psr_knots, name='band_gp')] if band else []) + \
            ([signals.makegp_fftcov_band_alpha(psr, signals.powerlaw, components=psr_knots, name='band_gp_alpha')] if band_alpha else []))

def make_common_gps_fourier(psrs, common_components=30, max_cadence_days=14, red=True, dm=True, chrom=True, sw=True, band=True, band_alpha=False):
    Tspan = signals.getspan(psrs)
    compound_gp = [matrix.CompoundGP(make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, red=red, dm=dm, chrom=chrom, sw=sw, band=band, band_alpha=band_alpha) +
                                     [signals.makegp_fourier(psr, signals.powerlaw, common_components, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')])
                   for psr in psrs]
    if not chrom and not band and not band_alpha:  # Static Fs, so we can use gps2commongp
       return gps2commongp(compound_gp)
    else:
        return compound_gp  # Does not work yet

def make_common_gps_fftint(psrs, common_knots=61, max_cadence_days=14, red=True, dm=True, chrom=True, sw=True, band=True, band_alpha=False):
    Tspan = signals.getspan(psrs)
    compound_gp = [matrix.CompoundGP(make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, red=red, dm=dm, chrom=chrom, sw=sw, band=band, band_alpha=band_alpha) +
                                     [signals.makegp_fftcov(psr, signals.powerlaw, common_knots, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')])
                   for psr in psrs]
    if not chrom and not band and not band_alpha: # Static Fs, so we can use gps2commongp
        return gps2commongp(compound_gp)
    else:
        return compound_gp  # Does not work yet


def single_pulsar_noise(psr, fftint=True, max_cadence_days=14, red=True, dm=True, chrom=True, sw=True, band=True, band_alpha=False):
    psr_Tspan = signals.getspan(psr)
    psr_components = int(psr_Tspan / (max_cadence_days * 86400))
    psr_knots = 2 * psr_components + 1
    noise = signals.makenoise_measurement(psr, tnequad=True)
    noise_params = getattr(noise, 'params', [])
    if fftint:
        model_components = [
            psr.residuals,
            noise,
            signals.makegp_ecorr(psr),
            signals.makegp_timing(psr, svd=True)]
        if red:
            model_components += [signals.makegp_fftcov(psr, signals.powerlaw, components=psr_knots, name='red_noise')]
        if dm:
            model_components += [signals.makegp_fftcov_dm(psr, signals.powerlaw, components=psr_knots, name='dm_gp')]
        if chrom:
            model_components += [signals.makegp_fftcov_chrom(psr, signals.powerlaw, components=psr_knots, name='chrom_gp')]
        if sw:
            model_components += [signals.makegp_fftcov_solar(psr, signals.powerlaw, components=psr_knots, name='sw_gp')]
        if band:
            model_components += [signals.makegp_fftcov_band(psr, signals.powerlaw, components=psr_knots, name='band_gp')]
        if band_alpha:
            model_components += [signals.makegp_fftcov_band_alpha(psr, signals.powerlaw, components=psr_knots, name='band_gp_alpha')]
    else:
        model_components = [
            psr.residuals,
            noise,
            signals.makegp_ecorr(psr),
            signals.makegp_timing(psr, svd=True)]
        if red:
            model_components += [signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, name='red_noise')]
        if dm:
            model_components += [signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_dm, name='dm_gp')]
        if chrom:
            model_components += [signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_chrom, name='chrom_gp')]
        if sw:
            model_components += [signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=solar.fourierbasis_solar_dm, name='sw_gp')]
        if band:
            model_components += [signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_low, name='band_gp')]
        if band_alpha:
            model_components += [signals.makegp_fourier(psr, signals.powerlaw, components=psr_components, fourierbasis=signals.fourierbasis_band_low_alpha, name='band_gp_alpha')]

    m = likelihood.PulsarLikelihood(model_components)
    m.all_params.extend(noise_params)
    m.logL.params = sorted(set(m.all_params))

    return m

