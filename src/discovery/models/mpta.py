import numpy as np

from .. import matrix
from .. import signals
from .. import prior
from .. import solar
from .. import likelihood
from .. import deterministic

prior.priordict_standard.update({
    # White noise parameters
    '(.*_)?efac':               [0.5, 2],
    '(.*_)?log10_tnequad':      [-10, -5],
    '(.*_)?log10_ecorr':        [-10, -5],
    # GP parameters
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
    '(.*_)?band_gp_alpha':      [0, 7],
    '(.*_)?band_gp_fcutoff':    [856, 1712],
    # common noise parameters
    'curn_log10_A':             [-18, -11],
    'curn_gamma':               [0, 7],
    # deterministic parameters
    '(.*_)?chrom_exp_t0': [58525, 60700],
    '(.*_)?chrom_exp_log10_Amp': [-10, -5],
    '(.*_)?chrom_exp_log10_tau': [0, 4],
    '(.*_)?chrom_exp_sign_param': [-1, 1],
    '(.*_)?chrom_exp_alpha': [0, 7],
    '(.*_)?chrom_1yr_log10_Amp': [-10, -5],
    '(.*_)?chrom_1yr_phase': [0, 2 * np.pi],
    '(.*_)?chrom_1yr_alpha': [0, 7],
    '(.*_)?chrom_gauss_t0': [58525, 60700],
    '(.*_)?chrom_gauss_log10_Amp': [-10, -5],
    '(.*_)?chrom_gauss_log10_sigma': [0, 4],
    '(.*_)?chrom_gauss_sign_param': [-1, 1],
    '(.*_)?chrom_gauss_alpha': [0, 7],
    r'(.*_)?timingmodel_coefficients\(\d+\)': [-20.0, 20.0],
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
    if not chrom and not band and not band_alpha:  # Static Fs, so we can use gps2commongp
       return gps2commongp([matrix.CompoundGP(make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, red=red, dm=dm, chrom=chrom, sw=sw, band=band, band_alpha=band_alpha) +
                                              [signals.makegp_fourier(psr, signals.powerlaw, common_components, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')])
                            for psr in psrs])
    else:
        return # Does not work yet

def make_common_gps_fftint(psrs, common_knots=61, max_cadence_days=14, red=True, dm=True, chrom=True, sw=True, band=True, band_alpha=False):
    Tspan = signals.getspan(psrs)
    if not chrom and not band and not band_alpha: # Static Fs, so we can use gps2commongp
        return gps2commongp([matrix.CompoundGP(make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, red=red, dm=dm, chrom=chrom, sw=sw, band=band, band_alpha=band_alpha) +
                                               [signals.makegp_fftcov(psr, signals.powerlaw, common_knots, Tspan, common=['curn_log10_A', 'curn_gamma'], name='curn')])
                            for psr in psrs]) # Does not work yet
    else:
        return # Does not work yet

def single_pulsar_noise(psr, noisedict, fftint=True, max_cadence_days=14, tm_variable=False, timing_inds=None,
                        wn=True, red=True, dm=True, chrom=True, sw=True, band=True, band_alpha=False, # GP models
                        chrom_annual=False, chrom_exponential=False, chrom_gaussian=False): # Deterministic chromatic models
    # Set up white noise
    if wn:
        measurement_noise = signals.makenoise_measurement(psr, tnequad=True)
    else:
        measurement_noise = signals.makenoise_measurement(psr, noisedict, tnequad=True)
    # Set up timing model
    tm = signals.makegp_timing(psr, svd=True, variable=tm_variable, timing_inds=timing_inds)  # ensure the timing model is unpacked if returning a list
    if not isinstance(tm, list):
        tm = [tm]
    # Set up model components
    model_components = [psr.residuals]
    model_components += tm
    model_components += [measurement_noise]
    if wn:
        model_components += [signals.makegp_ecorr(psr)]
    else:
        model_components += [signals.makegp_ecorr(psr, noisedict)]
    if chrom_annual:
        model_components += [signals.makedelay(psr, deterministic.chromatic_annual(psr), name='chrom_1yr')]
    if chrom_exponential:
        model_components += [signals.makedelay(psr, deterministic.chromatic_exponential(psr), name='chrom_exp')]
    if chrom_gaussian:
        model_components += [signals.makedelay(psr, deterministic.chromatic_gaussian(psr), name='chrom_gauss')]
    # Add GPS components
    if fftint:
        model_components += make_psr_gps_fftint(psr, max_cadence_days=max_cadence_days, red=red, dm=dm, chrom=chrom, sw=sw, band=band, band_alpha=band_alpha)
    else:
        model_components += make_psr_gps_fourier(psr, max_cadence_days=max_cadence_days, red=red, dm=dm, chrom=chrom, sw=sw, band=band, band_alpha=band_alpha)

    comp_params = []
    for comp in model_components:
        if hasattr(comp, 'params'):
            comp_params.extend(comp.params)

    m = likelihood.PulsarLikelihood(model_components)
    m.all_params.extend(comp_params)
    m.logL.params = sorted(set(m.all_params))

    return m

