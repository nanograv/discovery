from functools import partial
import discovery as ds
from typing import Optional, Callable


def make_array_likelihood(
    psrs: list[ds.Pulsar],
    correlation_orf: Optional[Callable]=None,
    spectrum: Callable=ds.powerlaw,
    intrinsic_red_components: int=30,
    common_components: int=14,
    gamma_common: Optional[float]=None
    ) -> ds.ArrayLikelihood:
    """Construct discovery array likelihood object from a list of Pulsar objects which may contain a noise dictionary.

    Parameters:
        psrs (list): List of pulsar objects
        correlation_orf (function, optional) [None]: Correlation overlap reduction function (ORF). Default gives intrinsic red noise only (IRN) model.
            Options include `ds.uncorrelated_orf`, `ds.hd_orf`, `ds.dipole_orf`, `ds.monopole_orf`, or a custom ORF function
        spectrum (function): Spectrum function for all red noise parameters
            Options include `ds.powerlaw`, `ds.freespectrum`, or a custom spectrum function
        intrinsic_red_components (int) [30]: Number of individual pulsar red noise components
        common_components (int) [14]: Number of common red noise components
        gamma_common (float) [None]: Common red noise spectral index

    Returns:
        array_likelihood (object): Discovery ArrayLikelihood object
    """

    tspan = ds.getspan(psrs)

    if gamma_common is not None and (correlation_orf is None or spectrum.__name__ != 'powerlaw'):
        raise ValueError('gamma_common is set but correlation_orf is None or spectrum is not a powerlaw.')

    # generate pulsar likelihood objects which are common to all ArrayLikelihood objects
    pulsar_likelihood_generator = (ds.PulsarLikelihood((psr.residuals,
                                   ds.makenoise_measurement(psr, psr.noisedict),
                                   ds.makegp_ecorr(psr, psr.noisedict),
                                   ds.makegp_timing(psr, svd=True),
                                   )) for psr in psrs)

    if correlation_orf is None:  # intrinsic red noise only

        if spectrum.__name__ == 'powerlaw':
            components = intrinsic_red_components
        elif spectrum.__name__ == 'freespectrum':
            components = {'log10_rho': intrinsic_red_components}
        else:
            raise ValueError('Power spectral density function not recognized.')

        print('Creating intrinsic red noise only model (IRN).')
        al = ds.ArrayLikelihood(pulsar_likelihood_generator,
                                ds.makecommongp_fourier(psrs, spectrum, components=components, T=tspan, name='red_noise'))

    elif correlation_orf.__name__ == 'uncorrelated_orf':  # CURN model

        if spectrum.__name__ == 'powerlaw':
            if gamma_common is not None:
                common_spectrum = ds.makepowerlaw_crn(common_components, crn_gamma=gamma_common)
                gamma_common_name = []
            else:
                common_spectrum = ds.makepowerlaw_crn(common_components)
                gamma_common_name = ['crn_gamma']

            common_names = ['crn_log10_A'] + gamma_common_name

        elif spectrum.__name__ == 'freespectrum':
            common_spectrum = ds.makefreespectrum_crn(common_components)
            components = {'log10_rho': intrinsic_red_components, 'crn_log10_rho': common_components}

        else:
            raise ValueError('Power spectral density function not recognized.')

        print('Creating uncorrelated red noise model (CURN).')
        al = ds.ArrayLikelihood(pulsar_likelihood_generator,
                                ds.makecommongp_fourier(psrs, common_spectrum, intrinsic_red_components, T=tspan, common=common_names, name='red_noise'))

    else:

        if spectrum.__name__ == 'powerlaw':
            if gamma_common is not None:
                common_spectrum = partial(spectrum, gamma=gamma_common)
                gamma_common_name = []
            else:
                common_spectrum = spectrum
                gamma_common_name = ['gw_gamma']

            common_names = ['gw_log10_A'] + gamma_common_name

            print(f'Creating {correlation_orf.__name__} correlated red noise model.')
            al = ds.ArrayLikelihood(pulsar_likelihood_generator,
                                    ds.makecommongp_fourier(psrs, spectrum, intrinsic_red_components, T=tspan, name='red_noise'),
                                    ds.makegp_fourier_global(psrs, spectrum, correlation_orf, common_components, T=tspan, name='gw'))

        # TODO: Implement free spectrum model for correlated models in ArrayLikelihood
        # elif spectrum.__name__ == 'freespectrum':
        #     common_spectrum = spectrum
        #     intrinsic_components = {'log10_rho': intrinsic_red_components}
        #     global_components = {'log10_rho': intrinsic_red_components, 'gw_log10_rho': common_components}

        #     print(f'Creating {correlation_orf.__name__} correlated red noise model.')
        #     al = ds.ArrayLikelihood(pulsar_likelihood_generator,
        #                             ds.makecommongp_fourier(psrs, spectrum, intrinsic_components, T=tspan, name='red_noise'),
        #                             ds.makegp_fourier_global(psrs, spectrum, correlation_orf, global_components, T=tspan, name='gw'))

        else:
            raise ValueError('Power spectral density function not recognized.')

    return al
