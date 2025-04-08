from .. import signals
from .. import likelihood


# Signle pulsar noise analysis. Data the second data relise of the European Pulsar Timing Array (DR2new+ dataset)
# Note: The exponential dips are not included in this model
def makemodel_singlepulsar(psrs, psr_name):

    for p in psrs:
        if psr_name in p.name:
            sgl_psr = p

    model = [sgl_psr.residuals, signals.makenoise_measurement(sgl_psr, sgl_psr.noisedict), signals.makegp_timing(sgl_psr)]

    if sgl_psr.noisedict[sgl_psr.name + '_dm_gp_components']: 
            model.append(signals.makegp_fourier(sgl_psr, signals.powerlaw, sgl_psr.noisedict[sgl_psr.name + '_dm_gp_components'], T=signals.getspan(sgl_psr), name='dm_gp', fourierbasis=signals.make_dmfourierbasis(alpha = 2.0, tndm = False)))

    if sgl_psr.noisedict[sgl_psr.name + '_chrom_components']: 
            model.append(signals.makegp_fourier(sgl_psr, signals.powerlaw, sgl_psr.noisedict[sgl_psr.name + '_chrom_components'], T=signals.getspan(sgl_psr), name='chrom_gp', fourierbasis=signals.make_dmfourierbasis(alpha = 4.0, tndm = False)))
        
    if sgl_psr.noisedict[sgl_psr.name + '_red_components']: 
            model.append(signals.makegp_fourier(sgl_psr, signals.powerlaw, sgl_psr.noisedict[sgl_psr.name + '_red_components'], T=signals.getspan(sgl_psr), name='red_noise' ))

    return likelihood.PulsarLikelihood(model)


# CURN model from the second data relise of the European Pulsar Timing Array (DR2new+ dataset).
# Note: The exponential dips are not included in this model
def makemodel_curn_EPTA(psrs, crn_components = 30):

    pslmodels = []
    tspan = signals.getspan(psrs)

    for p in psrs:

        model = [p.residuals, signals.makenoise_measurement(p, p.noisedict), signals.makegp_timing(p, svd=True)]

        if p.noisedict[p.name + '_dm_gp_components']: 
            model.append(signals.makegp_fourier(p, signals.powerlaw, p.noisedict[p.name + '_dm_gp_components'], T=signals.getspan(p), name='dm_gp', fourierbasis=signals.make_dmfourierbasis(alpha = 2.0, tndm = True)))

        if p.noisedict[p.name + '_chrom_components']: 
            model.append(signals.makegp_fourier(p, signals.powerlaw, p.noisedict[p.name + '_chrom_components'], T=signals.getspan(p), name='chrom_gp', fourierbasis=signals.make_dmfourierbasis(alpha = 4.0, tndm = True)))
        
        if p.noisedict[p.name + '_red_components']: 
            model.append(signals.makegp_fourier(p, signals.powerlaw, p.noisedict[p.name + '_red_components'], T=tspan, name='red_noise' ))

        pslmodels.append(likelihood.PulsarLikelihood(model))

    return likelihood.GlobalLikelihood(psls = pslmodels, globalgp=signals.makegp_fourier_global(psrs, signals.powerlaw, signals.uncorrelated_orf, components=crn_components, T=tspan, name='gw_crn'))


# HD model from the second data relise of the European Pulsar Timing Array (DR2new+ dataset).
# Note: The exponential dips are not included in this model
def makemodel_hd_EPTA(psrs, gw_components = 30):

    pslmodels = []
    tspan = signals.getspan(psrs)

    for p in psrs:

        model = [p.residuals, signals.makenoise_measurement(p, p.noisedict), signals.makegp_timing(p, svd=True)]

        if p.noisedict[p.name + '_dm_gp_components']: 
            model.append(signals.makegp_fourier(p, signals.powerlaw, p.noisedict[p.name + '_dm_gp_components'], T=signals.getspan(p), name='dm_gp', fourierbasis=signals.make_dmfourierbasis(alpha = 2.0, tndm = True)))

        if p.noisedict[p.name + '_chrom_components']: 
            model.append(signals.makegp_fourier(p, signals.powerlaw, p.noisedict[p.name + '_chrom_components'], T=signals.getspan(p), name='chrom_gp', fourierbasis=signals.make_dmfourierbasis(alpha = 4.0, tndm = True)))
        
        if p.noisedict[p.name + '_red_components']: 
            model.append(signals.makegp_fourier(p, signals.powerlaw, p.noisedict[p.name + '_red_components'], T=tspan, name='red_noise' ))

        pslmodels.append(likelihood.PulsarLikelihood(model))

    return likelihood.GlobalLikelihood(psls = pslmodels, globalgp=signals.makegp_fourier_global(psrs, signals.powerlaw, signals.hd_orf, components=gw_components, T=tspan, name='gw_hd'))

