import functools

import numpy as np
import scipy.integrate

from . import matrix
from . import signals
jnp = matrix.jnp

import jax

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from discovery import GlobalLikelihood, VariableGP, PulsarLikelihood

    from jax.typing import ArrayLike

    from collections.abc import Callable

# these versions of ORFs take only one parameter (the angle)
# z = jnp.dot(pos1, pos2)

def hd_orfa(z: ArrayLike) -> ArrayLike:
    """Calculate the Hellings and Downs overlap reduction function.

    Parameters
    ----------
    z : ArrayLike
        The cosine(s) of the angle(s) between two pulsars.

    Returns
    -------
    ArrayLike
        The Hellings and Downs correlation(s) for `z` .

    """
    omc2 = (1.0 - z) / 2.0
    return 1.5 * omc2 * jnp.log(omc2) - 0.25 * omc2 + 0.5 + 0.5 * jnp.allclose(z, 1.0)

def dipole_orfa(z: ArrayLike) -> ArrayLike:
    """Calculate the dipole overlap reduction function.

    Parameters
    ----------
    z : ArrayLike
        The cosine(s) of the angle(s) between two pulsars.

    Returns
    -------
    ArrayLike
        The dipole correlation(s) evaluated at `z`.

    """
    return z + 1.0e-6 * jnp.allclose(z, 1.0)

def monopole_orfa(z: ArrayLike) -> ArrayLike:
    """Calculate the monopole overlap reduction function.

    Parameters
    ----------
    z : ArrayLike
        The cosine(s) of the angle(s) between two pulsars.

    Returns
    -------
    ArrayLike
        The monopole correlation for `z`.

    """
    return 1.0 + 1.0e-6 * jnp.allclose(z, 1.0)


class OS:
    """Optimal Statistic (OS) class.

    This class implements various methods for calculating and analyzing the optimal statistic.

    Attributes
    ----------
    psls : list[PulsarLikelihood]
        List of PulsarLikelihood objects.
    gws : list[VariableGP]
        List of GW GP objects for each pulsar.
    gwpar : str
        Name of the GW amplitude parameter.
    pos : list[jax.Array]
        List of pulsar position vectors.
    pairs : list[tuple[int, int]]
        List of pulsar pair indices.
    angles : list[jax.Array]
        List of angles between pulsar pairs.
    params : list[str]
        List of parameters used in calculation.
    os_rhosigma : Callable[[dict[str, float]], tuple[jax.Array, jax.Array]]
        Cached closure that takes a dictionary of parameters and retunrs the cross correlations and correpsonding
        uncertainties
    os :

    """

    def __init__(self, gbl: GlobalLikelihood):
        """Initialize the Optimal Statistic object.

        Parameters
        ----------
        gbl : GlobalLikelihood
            A GlobalLikelihood object containing pulsar likelihoods.

        Raises
        ------
        AttributeError
            If the common GW GP cannot be found in the pulsar likelihood objects.

        """
        self.psls = gbl.psls

        try:
            self.gws = [psl.gw for psl in self.psls]
            self.gwpar = [par for par in self.gws[0].gpcommon if 'log10_A' in par][0]
            self.pos = [matrix.jnparray(psl.gw.pos) for psl in self.psls]
        except AttributeError:
            raise AttributeError("I cannot find the common GW GP in the pulsar likelihood objects.")

        self.pairs = [(i1, i2) for i1 in range(len(self.pos)) for i2 in range(i1 + 1, len(self.pos))]
        self.angles = [jnp.dot(self.pos[i], self.pos[j]) for (i,j) in self.pairs]

    @functools.cached_property
    def params(self) -> list[str]:
        """
        Get the parameters used in the optimal statistic calculation.

        Returns
        -------
        list[str]
            A list of parameter names.

        """
        return self.os_rhosigma.params

    @functools.cached_property
    def os_rhosigma(self) -> Callable[[dict[str, float]], tuple[jax.Array, jax.Array]]:
        """Calculate the cross-correlation values (rho) and their uncertainties (sigma).

        Returns
        -------
        get_rhosigma : Callable[[dict[str, float]], tuple[jax.Array, jax.Array]]
            A function that computes rho and sigma given a set of parameters.

        """
        kernelsolves = [psl.N.make_kernelsolve(psl.y, gw.F) for (psl, gw) in zip(self.psls, self.gws)]
        getN = self.gws[0].Phi.getN   # use prior from first pulsar, assume all GW GP are the same
        pairs = self.pairs

        def get_rhosigma(params: dict[str, float]) -> tuple[jax.Array, jax.Array]:
            """Compute rho and sigma for given parameters.

            Parameters
            ----------
            params : dict[str, float]
                Dictionary of parameter names and values.

            Attributes
            ----------
            params : set
                Sorted set of kernel parameters.

            Returns
            -------
            tuple[jax.Array, jax.Array]
                Tuple containing rho and sigma arrays.

            """
            sN = jnp.sqrt(getN(params))
            ks = [k(params) for k in kernelsolves]

            ts = [jnp.dot(sN * ks[i][0], sN * ks[j][0]) for (i, j) in pairs]

            ds = [sN[:, jnp.newaxis] * k[1] * sN[jnp.newaxis, :] for k in ks]
            bs = [jnp.trace(ds[i] @ ds[j]) for (i, j) in pairs]

            return (matrix.jnparray(ts) / matrix.jnparray(bs), 1.0 / jnp.sqrt(matrix.jnparray(bs)))

        get_rhosigma.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_rhosigma

    @functools.cached_property
    def os(self) -> Callable[dict[str, float]]:
        """Calculate the optimal statistic.

        Returns
        -------
        Callable
            A function that computes the optimal statistic, its uncertainty,
            signal-to-noise ratio, and log10 of the GW amplitude given parameters
            and an optional overlap reduction function (ORF).

        """
        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        gwpar, angles = self.gwpar, matrix.jnparray(self.angles)

        def get_os(params: dict[str, float], orf: Callable[[ArrayLike], ArrayLike]=hd_orfa):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            orfs = orf(angles)

            os = jnp.sum(rhos * orfs / sigmas**2) / jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / jnp.sqrt(jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} # , 'rhos': rhos, 'sigmas': sigmas}

        get_os.params = os_rhosigma.params

        return get_os

    @functools.cached_property
    def scramble(self) -> Callable:
        """Calculate the scrambled optimal statistic.

        Returns
        -------
        get_scramble
            A function that computes the scrambled optimal statistic given
            parameters, pulsar positions, and an optional ORF.

        """
        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        gwpar, pairs = self.gwpar, self.pairs

        def get_scramble(
            params: dict[str, float], pos: list[jax.Array], orf: Callable[[ArrayLike], ArrayLike] = hd_orfa
        ) -> dict[str, float]:
            """Compute the scrambled optimal statistic.

            Parameters
            ----------
            params : dict[str, float]
                Dictionary of parameter names and values.
            pos : list[jax.Array]
                List of pulsar position vectors.
            orf : Callable, optional
                Overlap reduction function, by default hd_orfa.

            Returns
            -------
            dict[str, float]
                Dictionary containing 'os', 'os_sigma', 'snr', and 'log10_A'.

            """
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10 ** (2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            angles = matrix.jnparray([jnp.dot(pos[i], pos[j]) for (i, j) in pairs])
            orfs = orf(angles)

            os = jnp.sum(rhos * orfs / sigmas**2) / jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / jnp.sqrt(jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {"os": os, "os_sigma": os_sigma, "snr": snr, "log10_A": params[gwpar]}  # , 'rhos': rhos, 'sigmas': sigmas}

        get_scramble.params = os_rhosigma.params

        return get_scramble

    @functools.cached_property
    def os_rhosigma_complex(self) -> Callable[[dict[str,float]], tuple[jax.Array, jax.Array]]:
        kernelsolves = [psl.N.make_kernelsolve(psl.y, gw.F) for (psl, gw) in zip(self.psls, self.gws)]
        getN = self.gws[0].Phi.getN
        pairs = self.pairs

        def get_rhosigma_complex(params):
            sN = jnp.sqrt(getN(params))
            ks = [k(params) for k in kernelsolves]

            tsf = [sN[::2] * (k[0][::2] + 1j * k[0][1::2]) for k in ks]
            ts = [tsf[i] * jnp.conj(tsf[j]) for (i,j) in pairs]

            ds = [sN[:,jnp.newaxis] * k[1] * sN[jnp.newaxis,:] for k in ks]
            bs = [jnp.trace(ds[i] @ ds[j]) for (i,j) in pairs]

            # can't use matrix.jnparray or complex will be downcast
            return (jnp.array(ts) / matrix.jnparray(bs)[:,jnp.newaxis],
                    1.0 / jnp.sqrt(matrix.jnparray(bs)))

        get_rhosigma_complex.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_rhosigma_complex

    @functools.cached_property
    def shift(self):
        os_rhosigma_complex = self.os_rhosigma_complex    # getos will close on os_rhosigma
        gwpar, pairs, angles = self.gwpar, self.pairs, matrix.jnparray(self.angles)

        def get_shift(params, phases, orf=hd_orfa):
            rhos_complex, sigmas = os_rhosigma_complex(params)

            # can't use matrix.jnparray or complex will be downcast
            phaseprod = jnp.array([jnp.exp(1j * (phases[i] - phases[j])) for i,j in pairs])
            rhos = jnp.sum(jnp.real(rhos_complex * phaseprod), axis=1)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            orfs = orf(angles)

            os = jnp.sum(rhos * orfs / sigmas**2) / jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / jnp.sqrt(jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} #, 'rhos': rhos, 'sigmas': sigmas}

        get_shift.params = os_rhosigma_complex.params

        return get_shift

    @functools.cached_property
    def gx2eig(self):
        kernelsolves = [psl.N.make_kernelsolve(psl.N.F, gw.F) for (psl, gw) in zip(self.psls, self.gws)]
        phis = [psr.N.P_var.getN for psr in self.psls]
        getN = self.gws[0].Phi.getN

        orfmat = matrix.jnparray([[signals.hd_orf(p1, p2) for p1 in self.pos] for p2 in self.pos])
        gwpar, pairs, orfs = self.gwpar, self.pairs, [signals.hd_orf(self.pos[i], self.pos[j]) for i, j in self.pairs]

        def get_gx2eig(params):
            sN = jnp.sqrt(getN(params))
            ks = [k(params) for k in kernelsolves]

            A = 10**params[gwpar]

            ts = [sN[:,jnp.newaxis] * k[0] * jnp.sqrt(phi(params)) / A for k, phi in zip(ks, phis)]

            d = [sN[:,jnp.newaxis] * k[1] * sN[jnp.newaxis,:] / A**2 for k in ks]
            b = sum(jnp.trace(d[i] @ d[j] * orf**2) for (i,j), orf in zip(pairs, orfs))

            amat = jnp.block([[(0.0 if i == j else orfmat[i,j] / jnp.sqrt(b)) * jnp.dot(t1.T, t2)
                               for i,t1 in enumerate(ts)]
                              for j,t2 in enumerate(ts)])

            return jnp.real(jnp.linalg.eig(amat)[0])

        get_gx2eig.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_gx2eig

    @functools.cached_property
    def imhof(self):
        def get_imhof(u, x, eigs):
            theta = 0.5 * jnp.sum(jnp.arctan(eigs[:,jnp.newaxis] * u), axis=0) - 0.5 * x * u
            rho = jnp.prod((1.0 + (eigs[:,jnp.newaxis] * u)**2)**0.25, axis=0)

            return jnp.sin(theta) / (u * rho)

        return jax.jit(get_imhof)

    # note this returns a numpy array, and the integration is handled by non-jax scipy
    def gx2cdf(self, params, xs, cutoff=1e-6, limit=100, epsabs=1e-6):
        eigr = self.gx2eig(params)

        # cutoff by number of eigenvalues is more friendly to jitted imhof
        eigs = eigr[:cutoff] if cutoff > 1 else eigr[jnp.abs(eigr) > cutoff]

        return np.array([0.5 - scipy.integrate.quad(lambda u: float(self.imhof(u, x, eigs)),
                                                    0, np.inf, limit=limit, epsabs=epsabs)[0] / np.pi for x in xs])
