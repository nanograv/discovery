import functools

import numpy as np
import scipy.integrate

from . import matrix
from . import signals

import jax

# these versions of ORFs take only one parameter (the angle)
# z = matrix.jnp.dot(pos1, pos2)

def hd_orfa(z):
    omc2 = (1.0 - z) / 2.0
    return 1.5 * omc2 * matrix.jnp.log(omc2) - 0.25 * omc2 + 0.5 + 0.5 * matrix.jnp.allclose(z, 1.0)

def dipole_orfa(z):
    return z + 1.0e-6 * matrix.jnp.allclose(z, 1.0)

def monopole_orfa(z):
    return 1.0 + 1.0e-6 * matrix.jnp.allclose(z, 1.0)


class OS:
    def __init__(self, gbl):
        self.psls = gbl.psls

        try:
            self.gws = [psl.gw for psl in self.psls]
            self.gwpar = [par for par in self.gws[0].gpcommon if 'log10_A' in par][0]
            self.pos = [matrix.jnparray(psl.gw.pos) for psl in self.psls]
        except AttributeError:
            raise AttributeError("I cannot find the common GW GP in the pulsar likelihood objects.")

        self.pairs = [(i1, i2) for i1 in range(len(self.pos)) for i2 in range(i1 + 1, len(self.pos))]
        self.angles = [matrix.jnp.dot(self.pos[i], self.pos[j]) for (i,j) in self.pairs]

    @functools.cached_property
    def params(self):
        return self.os_rhosigma.params

    @functools.cached_property
    def os_rhosigma(self):
        kernelsolves = [psl.N.make_kernelsolve(psl.y, gw.F) for (psl, gw) in zip(self.psls, self.gws)]
        getN = self.gws[0].Phi.getN   # use GW prior from first pulsar, assume all GW GP are the same
        pairs = self.pairs

        # OS = sum_{i<j} y_i* K_i^{-1} T_i Phi_{ij} T_j* K_j^{-1} y_j /
        #      sum_{j<j} tr K_i^{-1} T_i Phi_{ij} T_j* K_j^{-1} T_j Phi_{ji} T_i*
        #
        # with Phi_{ij} = orf_ij Phi
        #
        # kernelsolves return kv_i = T_i* K_i^{-1} y_i and km_i = T_i* K_i^{-1} T_i
        # and U* U = Phi
        #
        # then ts_ij = (U kv_i)* (U kv_j) and bs_ij = tr(U km_i U*  U km_j U*)
        #      rho_ij = ts_ij / bs_ij and sigma_ij = 1.0 / sqrt(bs_ij)
        #
        # and finally os = sum_{i<j} (ts_ij orf_ij) / sum_{i<j} (orf_ij^2 bs_ij)
        #                = sum_{i<j} (rho_ij orf_ij / sigma_ij^2) / sum_{i<j} (orf_ij^2 / sigma_ij^2)
        # and os_sigma   = (sum_{i<j} orf_ij^2 bs_ij)^(-1/2) = (sum_{i<j} orf_ij^2 / sigma_ij^2)^(-1/2)

        def get_rhosigma(params):
            N = getN(params)
            ks = [k(params) for k in kernelsolves]

            if N.ndim == 1:
                sN = matrix.jnp.sqrt(N)

                ts = [matrix.jnp.dot(sN * ks[i][0], sN * ks[j][0]) for (i,j) in pairs]
                ds = [sN[:,matrix.jnp.newaxis] * k[1] * sN[matrix.jnp.newaxis,:] for k in ks]

                bs = [matrix.jnp.trace(ds[i] @ ds[j]) for (i,j) in pairs]
            else:
                U = matrix.jnp.linalg.cholesky(N, upper=True) # N = U^T U

                uks = [U @ k[0] for k in ks]
                ds = [U @ k[1] @ U.T for k in ks]

                ts  = [matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in pairs]
                bs = [matrix.jnp.trace(ds[i] @ ds[j]) for (i,j) in pairs]

                # slower:
                # ts = [matrix.jnp.dot(U @ ks[i][0], U @ ks[j][0]) for (i,j) in pairs]
                # even slower, more explicit:
                # ts = [ks[i][0].T @ N @ ks[j][0] for (i,j) in pairs]

                # more explicit:
                # bs = [matrix.jnp.trace(ks[i][1] @ N @ ks[j][1] @ N) for (i,j) in pairs]

            return (matrix.jnparray(ts) / matrix.jnparray(bs),
                    1.0 / matrix.jnp.sqrt(matrix.jnparray(bs)))

        get_rhosigma.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_rhosigma

    @functools.cached_property
    def os(self):
        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        gwpar, angles = self.gwpar, matrix.jnparray(self.angles)

        def get_os(params, orf=hd_orfa):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            orfs = orf(angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} # , 'rhos': rhos, 'sigmas': sigmas}

        get_os.params = os_rhosigma.params

        return get_os

    @functools.cached_property
    def scramble(self):
        os_rhosigma = self.os_rhosigma    # getos will close on os_rhosigma
        gwpar, pairs = self.gwpar, self.pairs

        def get_scramble(params, pos, orf=hd_orfa):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            angles = matrix.jnparray([matrix.jnp.dot(pos[i], pos[j]) for (i,j) in pairs])
            orfs = orf(angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'log10_A': params[gwpar]} #, 'rhos': rhos, 'sigmas': sigmas}

        get_scramble.params = os_rhosigma.params

        return get_scramble

    @functools.cached_property
    def os_rhosigma_complex(self):
        kernelsolves = [psl.N.make_kernelsolve(psl.y, gw.F) for (psl, gw) in zip(self.psls, self.gws)]
        getN = self.gws[0].Phi.getN
        pairs = self.pairs

        def get_rhosigma_complex(params):
            N = getN(params)
            ks = [k(params) for k in kernelsolves]

            if sN.ndim == 2:
                raise NotImplementedError("Complex rhosigma not defined for 2D Phi.")

            sN = matrix.jnp.sqrt(N)

            tsf = [sN[::2] * (k[0][::2] + 1j * k[0][1::2]) for k in ks]
            ts = [tsf[i] * matrix.jnp.conj(tsf[j]) for (i,j) in pairs]

            ds = [sN[:,matrix.jnp.newaxis] * k[1] * sN[matrix.jnp.newaxis,:] for k in ks]
            bs = [matrix.jnp.trace(ds[i] @ ds[j]) for (i,j) in pairs]

            # can't use matrix.jnparray or complex will be downcast
            return (matrix.jnparray(ts) / matrix.jnparray(bs)[:,matrix.jnp.newaxis],
                    1.0 / matrix.jnp.sqrt(matrix.jnparray(bs)))

        get_rhosigma_complex.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_rhosigma_complex

    @functools.cached_property
    def shift(self):
        os_rhosigma_complex = self.os_rhosigma_complex    # getos will close on os_rhosigma
        gwpar, pairs, angles = self.gwpar, self.pairs, matrix.jnparray(self.angles)

        def get_shift(params, phases, orf=hd_orfa):
            rhos_complex, sigmas = os_rhosigma_complex(params)

            # can't use matrix.jnparray or complex will be downcast
            phaseprod = matrix.jnp.array([matrix.jnp.exp(1j * (phases[i] - phases[j])) for i,j in pairs])
            rhos = matrix.jnp.sum(matrix.jnp.real(rhos_complex * phaseprod), axis=1)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            orfs = orf(angles)

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
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
            N = getN(params)
            ks = [k(params) for k in kernelsolves]

            A = 10**params[gwpar]

            if N.ndim == 1:
                sN = matrix.jnp.sqrt(N)

                ts = [sN[:,matrix.jnp.newaxis] * k[0] * matrix.jnp.sqrt(phi(params)) / A for k, phi in zip(ks, phis)]
                ds = [sN[:,matrix.jnp.newaxis] * k[1] * sN[matrix.jnp.newaxis,:] / A**2 for k in ks]
            else:
                U = matrix.jnp.linalg.cholesky(N, upper=True)

                ts = [U @ k[0] @ matrix.jnp.linalg.cholesky(phi(params), upper=True) / A for k, phi in zip(ks, phis)]
                ds = [U @ k[1] * U.T / A**2 for k in ks]

            b = sum(matrix.jnp.trace(ds[i] @ ds[j] * orf**2) for (i,j), orf in zip(pairs, orfs))

            amat = matrix.jnp.block([[(0.0 if i == j else orfmat[i,j] / matrix.jnp.sqrt(b)) * matrix.jnp.dot(t1.T, t2)
                               for i, t1 in enumerate(ts)]
                              for j, t2 in enumerate(ts)])

            return matrix.jnp.real(matrix.jnp.linalg.eig(amat)[0])

        get_gx2eig.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

        return get_gx2eig

    @functools.cached_property
    def imhof(self):
        def get_imhof(u, x, eigs):
            theta = 0.5 * matrix.jnp.sum(matrix.jnp.arctan(eigs * u), axis=0) - 0.5 * x * u
            rho = matrix.jnp.prod((1.0 + (eigs * u)**2)**0.25, axis=0)

            return matrix.jnp.sin(theta) / (u * rho)

        return jax.jit(get_imhof)

    # note this returns a numpy array, and the integration is handled by non-jax scipy
    def gx2cdf(self, params, xs, cutoff=1e-6, limit=100, epsabs=1e-6):
        eigr = self.gx2eig(params)

        # cutoff by number of eigenvalues is more friendly to jitted imhof
        eigs = eigr[:cutoff] if cutoff > 1 else eigr[matrix.jnp.abs(eigr) > cutoff]

        return np.array([0.5 - scipy.integrate.quad(lambda u: float(self.imhof(u, x, eigs)),
                                                    0, np.inf, limit=limit, epsabs=epsabs)[0] / np.pi for x in xs])
