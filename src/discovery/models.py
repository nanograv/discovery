from functools import partial
import discovery as ds
from typing import Optional, Callable


def make_array_likelihood(
    psrs: list[ds.Pulsar],
    gamma_common: Optional[float]=None,
    correlation_orf: Optional[Callable]=None,
    intrinsic_red_components: int=30,
    common_components: int=14,
    spectrum: Callable=ds.powerlaw,
    ) -> ds.ArrayLikelihood:
    """Construct discovery array likelihood object from a list of Pulsar objects which may contain a noise dictionary.

    Parameters:
        psrs (list): List of pulsar objects
        gamma_common (float) [None]: Common red noise spectral index
        correlation_orf (function, optional) [None]: Correlation overlap reduction function (ORF). Default gives intrinsic red noise only (IRN) model.
            Options include `ds.uncorrelated_orf`, `ds.hd_orf`, `ds.dipole_orf`, `ds.monopole_orf`, or a custom ORF function
        intrinsic_red_components (int) [30]: Number of individual pulsar red noise components
        common_components (int) [14]: Number of common red noise components
        spectrum (function): Spectrum function for all red noise parameters
            Options include `ds.powerlaw`, `ds.freespectrum`, or a custom spectrum function
        tnequad (bool): Use temponest measurement noise definition

    Returns:
        array_likelihood (object): Discovery ArrayLikelihood object
    """

    tspan = ds.getspan(psrs)

    if spectrum.__name__ == 'powerlaw':  # powerlaw pieces
        if gamma_common is not None and correlation_orf is None:
            print('Warning: gamma_common is set but correlation_orf is None. Setting gamma_common to None.')
            gamma_common_name = []
            common_spectrum = None  # just use spectrum in this case
        elif gamma_common is not None:
            print(f'Creating powerlaw spectrum red noise model with gamma={gamma_common}.')
            common_spectrum = ds.makepowerlaw_crn(common_components, crn_gamma=gamma_common)
            gamma_common_name = []
        else:
            print('Creating powerlaw spectrum red noise model with varied gamma.')
            common_spectrum = ds.makepowerlaw_crn(common_components)
            gamma_common_name = ['crn_gamma']

        common_names = ['crn_log10_A'] + gamma_common_name
        components = intrinsic_red_components

    elif spectrum.__name__ == 'freespectrum':  # free spectrum pieces
        if correlation_orf is None:
            common_spectrum = ds.makefreespectrum_crn(common_components)
            components = {'log10_rho': intrinsic_red_components}
        else:
            common_spectrum = ds.makefreespectrum_crn(common_components)
            components = {'log10_rho': intrinsic_red_components, 'crn_log10_rho': common_components}

        common_names = ['crn_log10_rho']

    else:
        raise ValueError('Spectrum function not recognized.')

    # correlation block
    if correlation_orf is None:  # intrinsic red noise only
        print('Creating intrinsic red noise only model (IRN).')
        al = ds.ArrayLikelihood((ds.PulsarLikelihood((psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_ecorr(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True),
                                 )) for psr in psrs),
                                 ds.makecommongp_fourier(psrs, spectrum, components=components, T=tspan, name='red_noise'))
    elif correlation_orf.__name__ == 'uncorrelated_orf':
        print('Creating uncorrelated red noise model (CURN).')
        al = ds.ArrayLikelihood((ds.PulsarLikelihood((psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_ecorr(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True),
                                 )) for psr in psrs),
                                 ds.makecommongp_fourier(psrs, common_spectrum, intrinsic_red_components, T=tspan, common=common_names, name='red_noise'))
    else:
        print(f'Creating {correlation_orf.__name__} correlated red noise model.')
        al = ds.ArrayLikelihood((ds.PulsarLikelihood((psr.residuals,
                                 ds.makenoise_measurement(psr, psr.noisedict),
                                 ds.makegp_ecorr(psr, psr.noisedict),
                                 ds.makegp_timing(psr, svd=True))) for psr in psrs),
                                 ds.makecommongp_fourier(psrs, spectrum, intrinsic_red_components, T=tspan, name='red_noise'),
                                 ds.makegp_fourier_global(psrs, spectrum, correlation_orf, components, T=tspan, name='gw'))

    return al
