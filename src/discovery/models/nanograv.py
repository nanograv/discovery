from discovery import signals
from discovery import likelihood


# CURN model from 15yr NANOGrav analysis (= model 2a in 12.5yr analysis)
def makemodel_curn(psrs, rn_components=30, crn_components=14, gamma='variable'):
    Tspan = signals.getspan(psrs)

    pslmodels = [likelihood.PulsarLikelihood([psr.residuals,
                                              signals.makenoise_measurement(psr, psr.noisedict),
                                              signals.makegp_ecorr(psr, psr.noisedict),
                                              signals.makegp_timing(psr, svd=True)]) for psr in psrs]

    # note red_noise is backward-compatible with nanograv chains, "rednoise" is better in discovery
    curngp = signals.makecommongp_fourier(psrs, signals.makepowerlaw_crn(crn_components, crn_gamma=gamma), rn_components, T=Tspan,
                                          common=['crn_log10_A'] + (['crn_gamma'] if gamma == 'variable' else []),
                                          name='red_noise')

    return likelihood.ArrayLikelihood(pslmodels, commongp=curngp)


# HD model from 15yr NANOGrav analysis (= model 3a in 12.5yr analysis)
def makemodel_hd(psrs, rn_components=30, gw_components=14, gamma='variable'):
    Tspan = signals.getspan(psrs)

    pslmodels = [likelihood.PulsarLikelihood([psr.residuals,
                                              signals.makenoise_measurement(psr, psr.noisedict),
                                              signals.makegp_ecorr(psr, psr.noisedict),
                                              signals.makegp_timing(psr, svd=True)]) for psr in psrs]

    # note red_noise is backward-compatible with nanograv chains, "rednoise" is better in discovery
    rngp = signals.makecommongp_fourier(psrs, signals.powerlaw, rn_components, T=Tspan, name='red_noise')
    hdgp = signals.makeglobalgp_fourier(psrs, signals.powerlaw, signals.hd_orf, gw_components, T=Tspan, name='gw')

    return likelihood.ArrayLikelihood(pslmodels, commongp=rngp, globalgp=hdgp)
