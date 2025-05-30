import numpy as np

from discovery import matrix
from discovery import signals
from discovery import likelihood

priordict_standard = {
    "(.*_)?efac": [0.9, 1.1],
    "(.*_)?t2equad": [-8.5, -5],
    "(.*_)?tnequad": [-8.5, -5],
    "(.*_)?log10_ecorr": [-8.5, -5],
    "(.*_)?rednoise_log10_A.*": [-18, -10],
    "(.*_)?rednoise_gamma.*": [0, 7],
    "(.*_)?red_noise_log10_A.*": [-18, -10],  # deprecated
    "(.*_)?red_noise_gamma.*": [0, 7],  # deprecated
    "crn_log10_A.*": [-18, -11],
    "crn_gamma.*": [0, 7],
    "gw_(.*_)?log10_A": [-18, -11],
    "gw_(.*_)?gamma": [0, 7],
    "(.*_)?gp_log10_A": [-18, -10],
    "(.*_)?gp_gamma": [0, 7],
    "(.*_)?gp_alpha": [1, 3],
    "crn_log10_rho": [-9, -4],
    "gw_(.*_)?log10_rho": [-9, -4],
    r"(.*_)?red_noise_log10_rho\(([0-9]*)\)": [-9, -4],
    r"(.*_)?red_noise_crn_log10_rho\(([0-9]*)\)": [-9, -4],
    "(.*_)?log10_Amp": [-10, -2],
    "(.*_)?log10_tau": [0, 2.5],
    "(.*_)?2_t0": [54650, 54850],
    "(.*_)?1_t0": [57490, 57530]
}

# merge multiple GPs into a single CommonGP
# experimental, will be moved to signals.py
def gps2commongp(gps):
    priors = [gp.Phi.getN for gp in gps]
    pmax = len(gps)
    
    ns = [gp.F.shape[1] for gp in gps]
    nmax = max(ns)    
    
    def prior(params):
        yp = matrix.jnp.full((pmax, nmax), 1e-40)

        for i,p in enumerate(priors):
            yp = yp.at[i, :ns[i]].set(p(params))

        return yp
    prior.params = sorted(set([par for p in priors for par in p.params])) 
    
    Fs = [np.pad(gp.F, [(0,0), (0,nmax - gp.F.shape[1])]) for gp in gps]

    return matrix.VariableGP(matrix.VectorNoiseMatrix1D_var(prior), Fs)

# standard EPTA single-pulsar GPs
def _makegps(psr, Tred):
    return (([signals.makegp_fourier(psr, signals.powerlaw,
                                     components=psr.noisedict[psr.name + '_dm_gp_components'], T=signals.getspan(psr),
                                     fourierbasis=signals.make_dmfourierbasis(alpha=2.0, tndm=False), name='dm_gp')]
             if psr.noisedict[psr.name + '_dm_gp_components'] else []) +
            ([signals.makegp_fourier(psr, signals.powerlaw,
                                     components=psr.noisedict[psr.name + '_chrom_components'], T=signals.getspan(psr),
                                     fourierbasis=signals.make_dmfourierbasis(alpha=4.0, tndm=False), name='chrom_gp')]
             if psr.noisedict[psr.name + '_chrom_components'] else []) + 
            ([signals.makegp_fourier(psr, signals.powerlaw,
                                     components=psr.noisedict[psr.name + '_red_components'], T=Tred, name='red_noise')]
             if psr.noisedict[psr.name + '_red_components'] else []))

# single-pulsar noise analysis for EPTA DR2new+. No exponential dips
def makemodel_singlepulsar(psr):
    return likelihood.PulsarLikelihood([psr.residuals,
                                        signals.makenoise_measurement(psr, psr.noisedict),
                                        signals.makegp_timing(psr)] + _makegps(psr, Tred=signals.getspan(psr)))

# CURN model for EPTA DR2new+. No exponential dips
def makemodel_curn(psrs, crn_components=30, array=False):
    tspan = signals.getspan(psrs)

    if array:
        pslmodels = [likelihood.PulsarLikelihood([psr.residuals,
                                                  signals.makenoise_measurement(psr, psr.noisedict),
                                                  signals.makegp_timing(psr)])
                     for psr in psrs]

        cgp = gps2commongp([matrix.CompoundGP(_makegps(psr, tspan) +
                                              [signals.makegp_fourier(psr, signals.powerlaw, crn_components, tspan,
                                                                      common=['gw_crn_log10_A', 'gw_crn_gamma'], name='gw_crn')])
                            for psr in psrs])

        return likelihood.ArrayLikelihood(pslmodels, commongp=cgp)
    else:
        psls = [likelihood.PulsarLikelihood([psr.residuals,
                                             signals.makenoise_measurement(psr, psr.noisedict),
                                             signals.makegp_timing(psr)] + _makegps(psr, tspan) +
                                            [signals.makegp_fourier(psr, signals.powerlaw, crn_components, tspan,
                                                                    common=['gw_crn_log10_A', 'gw_crn_gamma'], name='gw_crn')])
                for psr in psrs]

        return likelihood.GlobalLikelihood(psls)

# HD model for EPTA DR2new+. No exponential dips
def makemodel_hd(psrs, gw_components=30, array=False):
    tspan = signals.getspan(psrs)

    if array:
        psls = [likelihood.PulsarLikelihood([psr.residuals,
                                            signals.makenoise_measurement(psr, psr.noisedict),
                                            signals.makegp_timing(psr)])
                for psr in psrs]

        cgp = gps2commongp([matrix.CompoundGP(_makegps(psr, tspan)) for psr in psrs])
        ggp = signals.makeglobalgp_fourier(psrs, signals.powerlaw, signals.hd_orf, components=gw_components, T=tspan, name='gw_hd')

        return likelihood.ArrayLikelihood(psls, commongp=cgp, globalgp=ggp)        
    else:
        psls = [likelihood.PulsarLikelihood([psr.residuals,
                                            signals.makenoise_measurement(psr, psr.noisedict),
                                            signals.makegp_timing(psr)] + _makegps(psr, tspan))
                for psr in psrs]

        ggp = signals.makeglobalgp_fourier(psrs, signals.powerlaw, signals.hd_orf, components=gw_components, T=tspan, name='gw_hd')

        return likelihood.GlobalLikelihood(psls, globalgp=ggp)

