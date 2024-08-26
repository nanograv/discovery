from jax.tree_util import Partial
import discovery as ds

def make_likelihood(psrs, noisedict=None, gamma_common=None,
                red_components=30, common_components=14,
                common_type='curn', array_like=False):
    """
    Construct discovery likelihood object from list of pulsar objects.
    Parameters:
    - psrs (list): List of pulsar objects.
    - noisedict (dict, optional): Dictionary containing noise properties for each pulsar. Defaults to None.
    - gamma_common (float, optional): Common red noise spectral index. Defaults to None.
    - red_components (int, optional): Number of red noise components. Defaults to 30.
    - common_components (int, optional): Number of common noise components. Defaults to 14.
    - common_type (str, optional): Type of common noise model. Defaults to 'curn'.
    - array_like (bool, optional): Whether to implement `batched` GPs (experimental). Defaults to False. [Not implemented yet]

    Returns:
    - gl (object): Discovery likelihood object.
    """

    tspan = ds.getspan(psrs)

    if gamma_common is not None:
        common_powerlaw = Partial(ds.powerlaw, gamma=gamma_common)
        gamma_common_name = []
    else:
        common_powerlaw = ds.powerlaw
        gamma_common_name = ['gw_gamma']

    if common_type == 'curn':
        gl = ds.GlobalLikelihood((ds.PulsarLikelihood([psr.residuals,
                                    ds.makenoise_measurement(psr, noisedict),
                                    ds.makegp_ecorr(psr, noisedict),
                                    ds.makegp_timing(psr, svd=True),
                                    ds.makegp_fourier(psr, ds.powerlaw, red_components, T=tspan, name='red_noise'),
                                    ds.makegp_fourier(psr, common_powerlaw, common_components, T=tspan,
                                                      common=['gw_log10_A']+gamma_common_name, name='gw')
                                                      ]) for psr in psrs))
    elif common_type == 'hd':
        gl = ds.GlobalLikelihood((ds.PulsarLikelihood([psr.residuals,
                                    ds.makenoise_measurement(psr, noisedict),
                                    ds.makegp_ecorr(psr, noisedict),
                                    ds.makegp_timing(psr, svd=True),
                                    ds.makegp_fourier(psr, ds.powerlaw, red_components, T=tspan, name="red_noise")
                                    ]) for psr in psrs),
                                    ds.makegp_fourier_global(psrs, common_powerlaw,
                                                             ds.hd_orf, common_components,
                                                             T=tspan, name="gw"))

    return gl