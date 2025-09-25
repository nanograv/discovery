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


def make2d(array):
    return matrix.jnp.diag(array) if array.ndim == 1 else array

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

    # TO DO: make opQ, and share init code between opQ, Q, and sample

    @functools.cached_property
    def Q(self):
        Nmats, Fmats, Tmats = zip(*[(psl.N.N.N, psl.N.F, psl.gw.F) for psl in self.psls])

        LNms = [1.0 / matrix.jnp.sqrt(Nmat) for Nmat in Nmats]
        Fts = [LNm[:,None] * Fmat for LNm, Fmat in zip(LNms, Fmats)]
        Tts = [LNm[:,None] * Tmat for LNm, Tmat in zip(LNms, Tmats)] # this is GW-only

        FFts = [matrix.jnparray(Ft.T @ Ft) for Ft in Fts]
        TTts = [matrix.jnparray(Tt.T @ Tt) for Tt in Tts]
        FTts = [matrix.jnparray(Ft.T @ Tt) for Ft, Tt in zip(Fts, Tts)]

        Phivar = self.psls[0].gw.Phi.getN
        Pvars = [psl.N.P_var.getN for psl in self.psls]

        ngw = Tts[0].shape[1]
        cnt = len(self.psls) * ngw
        inds = [slice(i * ngw, (i + 1) * ngw) for i in range(len(self.psls))]

        def get_Q(params, orf=hd_orfa):
            sPhi = matrix.jnp.sqrt(Phivar(params))

            cs = [matrix.jsp.linalg.cho_factor(matrix.jnp.diag(1.0 / Pvar(params)) + FFt) for Pvar, FFt in zip(Pvars, FFts)]
            Ss = [TTt - FTt.T @ matrix.jsp.linalg.cho_solve(c, FTt) for c, TTt, FTt in zip(cs, TTts, FTts)]

            Ss = [0.5 * (S + S.T) for S in Ss]  # ensure symmetry
            As = [matrix.jnp.linalg.cholesky(S + (1e-10 * matrix.jnp.trace(S) / S.shape[0]) * matrix.jnp.eye(S.shape[0]))
                for S in Ss]

            Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
            bs = [matrix.jnp.trace(Ds[i] @ Ds[j]) for (i,j) in self.pairs]

            orfs = orf(matrix.jnparray(self.angles))
            # note the 2 to get OS = x^T Q x
            denom = 2.0 * matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 * matrix.jnparray(bs)))

            Q = matrix.jnpzeros((cnt, cnt))

            A_scaled = [sPhi[:, None] * A for A in As]

            for w, (i, j) in zip(orfs, self.pairs):
                Bij = w * (A_scaled[i].T @ A_scaled[j])

                Q = Q.at[inds[i], inds[j]].add(Bij)
                Q = Q.at[inds[j], inds[i]].add(Bij.T)

            return Q / denom
        get_Q.params = self.os_rhosigma.params

        return get_Q

    @functools.cached_property
    def opQ(self):
        Nmats, Fmats, Tmats = zip(*[(psl.N.N.N, psl.N.F, psl.gw.F) for psl in self.psls])

        LNms = [1.0 / matrix.jnp.sqrt(Nmat) for Nmat in Nmats]
        Fts = [LNm[:,None] * Fmat for LNm, Fmat in zip(LNms, Fmats)]
        Tts = [LNm[:,None] * Tmat for LNm, Tmat in zip(LNms, Tmats)] # this is GW-only

        FFts = [matrix.jnparray(Ft.T @ Ft) for Ft in Fts]
        TTts = [matrix.jnparray(Tt.T @ Tt) for Tt in Tts]
        FTts = [matrix.jnparray(Ft.T @ Tt) for Ft, Tt in zip(Fts, Tts)]

        Phivar = self.psls[0].gw.Phi.getN
        Pvars = [psl.N.P_var.getN for psl in self.psls]

        ngw = Tts[0].shape[1]
        cnt = len(self.psls) * ngw
        inds = [slice(i * ngw, (i + 1) * ngw) for i in range(len(self.psls))]

        def get_opQ(params, orf=hd_orfa):
            sPhi = matrix.jnp.sqrt(Phivar(params))

            cs = [matrix.jsp.linalg.cho_factor(matrix.jnp.diag(1.0 / Pvar(params)) + FFt) for Pvar, FFt in zip(Pvars, FFts)]
            Ss = [TTt - FTt.T @ matrix.jsp.linalg.cho_solve(c, FTt) for c, TTt, FTt in zip(cs, TTts, FTts)]

            Ss = [0.5 * (S + S.T) for S in Ss]  # ensure symmetry
            As = [matrix.jnp.linalg.cholesky(S + (1e-10 * matrix.jnp.trace(S) / S.shape[0]) * matrix.jnp.eye(S.shape[0]))
                for S in Ss]

            Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
            bs = [matrix.jnp.trace(Ds[i] @ Ds[j]) for (i,j) in self.pairs]

            orfs = orf(matrix.jnparray(self.angles))
            # note the 2 to get OS = x^T Q x
            denom = 2.0 * matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 * matrix.jnparray(bs)))

            Bs = [sPhi[:, None] * A for A in As]   # B_i = diag(sPhi) @ A_i

            # currently not traceable; too bad
            def op(x):
                zs = [B @ x[ii] for B, ii in zip(Bs, inds)]

                y = matrix.jnp.zeros_like(x)
                for w, (i, j) in zip(orfs, self.pairs):
                    y = y.at[inds[i]].add((w / denom) * (Bs[i].T @ zs[j]))
                    y = y.at[inds[j]].add((w / denom) * (Bs[j].T @ zs[i]))

                return y

            return op
        get_opQ.params = self.os_rhosigma.params

        return get_opQ

    @functools.cached_property
    def sample(self):
        Nmats, Fmats, Tmats = zip(*[(psl.N.N.N, psl.N.F, psl.gw.F) for psl in self.psls])

        LNms = [1.0 / matrix.jnp.sqrt(Nmat) for Nmat in Nmats]
        Fts = [LNm[:,None] * Fmat for LNm, Fmat in zip(LNms, Fmats)]
        Tts = [LNm[:,None] * Tmat for LNm, Tmat in zip(LNms, Tmats)] # this is GW-only

        FFts = [matrix.jnparray(Ft.T @ Ft) for Ft in Fts]
        TTts = [matrix.jnparray(Tt.T @ Tt) for Tt in Tts]
        FTts = [matrix.jnparray(Ft.T @ Tt) for Ft, Tt in zip(Fts, Tts)]

        Phivar = self.psls[0].gw.Phi.getN
        Pvars = [psl.N.P_var.getN for psl in self.psls]

        ngw = Tts[0].shape[1]
        cnt = len(self.psls) * ngw
        inds = [slice(i * ngw, (i + 1) * ngw) for i in range(len(self.psls))]

        def get_sample(key, params, orf=hd_orfa):
            sPhi = matrix.jnp.sqrt(Phivar(params))

            # TO DO: should probably close on Ft.T @ Ft, Tt.T @ Tt, and Tt.T @ Ft (and Ft.T @ Tt) rather than on Fts and Tts
            cs = [matrix.jsp.linalg.cho_factor(matrix.jnp.diag(1.0 / Pvar(params)) + FFt) for Pvar, FFt in zip(Pvars, FFts)]
            Ss = [TTt - FTt.T @ matrix.jsp.linalg.cho_solve(c, FTt) for c, TTt, FTt in zip(cs, TTts, FTts)]

            Ss = [0.5 * (S + S.T) for S in Ss]  # ensure symmetry
            As = [matrix.jnp.linalg.cholesky(S + (1e-10 * matrix.jnp.trace(S) / S.shape[0]) * matrix.jnp.eye(S.shape[0]))
                  for S in Ss]

            Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
            bs = [matrix.jnp.trace(Ds[i] @ Ds[j]) for (i,j) in self.pairs]

            xs = matrix.jnpnormal(key, cnt)
            uks = [sPhi * (A @ xs[ind]) for A, ind in zip(As, inds)]

            ts = matrix.jnparray([matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in self.pairs])

            gwnorm = 10**(2.0 * params[self.gwpar])
            rhos = gwnorm * (matrix.jnparray(ts) / matrix.jnparray(bs))
            sigmas = gwnorm / matrix.jnp.sqrt(matrix.jnparray(bs))

            orfs = orf(matrix.jnparray(self.angles))

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return snr
        get_sample.params = self.os_rhosigma.params

        return get_sample

    def sample_rhosigma_lowrank(self, params, orf=hd_orfa):
        Phi = self.psls[0].gw.Phi.getN(params)
        sPhi = matrix.jnp.sqrt(Phi)

        Nmats, Fmats, Pmats, Tmats = zip(*[(psl.N.N.N, psl.N.F, psl.N.P_var.getN(params), psl.gw.F) for psl in self.psls])

        LNms = [1.0 / matrix.jnp.sqrt(Nmat) for Nmat in Nmats]
        Fts = [LNm[:,None] * Fmat for LNm, Fmat in zip(LNms, Fmats)]
        Tts = [LNm[:,None] * Tmat for LNm, Tmat in zip(LNms, Tmats)] # this is GW-only

        cs = [matrix.jsp.linalg.cho_factor(matrix.jnp.diag(1/Pmat) + Ft.T @ Ft) for Pmat, Ft in zip(Pmats, Fts)]
        Xs = [Tt - Ft @ matrix.jsp.linalg.cho_solve(c, Ft.T @ Tt) for c, Ft, Tt in zip(cs, Fts, Tts)]

        Ss = [Tt.T @ X for Tt, X in zip(Tts, Xs)]

        # alternative formulation (numerically unstable?):
        # R = chol(Pmat^-1 + Ft.T @ Ft)
        # Y = R^-1 @ Ft.T @ Tt
        # S = Tt.T @ Tt - Y.T @ Y
        #
        # Rs = [matrix.jnp.linalg.cholesky(matrix.jnp.diag(1/Pmat) + Ft.T @ Ft, upper=True) for Pmat, Ft in zip(Pmats, Fts)]
        # Ys = [matrix.jsp.linalg.solve_triangular(R, Ft.T @ Tt, lower=False) for R, Ft, Tt in zip(Rs, Fts, Tts)]
        # Ss = [Tt.T @ Tt - Y.T @ Y for Tt, Y in zip(Tts, Ys)]

        # with ridge regularization; the simple estimate based on the trace seems fine
        # a more precise possibility is eps = matrix.jnp.maximum(0.0, -matrix.jnp.linalg.eigvalsh(S).min())
        #                                     + 1e-10 * matrix.jnp.trace(S) / S.shape[0]
        Ss = [0.5 * (S + S.T) for S in Ss]  # ensure symmetry
        As = [matrix.jnp.linalg.cholesky(S + (1e-10 * matrix.jnp.trace(S) / S.shape[0]) * matrix.jnp.eye(S.shape[0]))
              for S in Ss]

        Ds = [sPhi[:,matrix.jnp.newaxis] * S * sPhi[matrix.jnp.newaxis,:] for S in Ss]
        bs = [matrix.jnp.trace(Ds[i] @ Ds[j]) for (i,j) in self.pairs]

        inds, cnt = [], 0
        for A in As:
            inds.append(slice(cnt, cnt + A.shape[0])) # these are all the same length, could simplify
            cnt += A.shape[0]

        def xs2snrs(xs):
            uks = [sPhi * (A @ xs[ind]) for A, ind in zip(As, inds)]

            ts = matrix.jnparray([matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in self.pairs])

            gwnorm = 10**(2.0 * params[self.gwpar])
            rhos = gwnorm * (matrix.jnparray(ts) / matrix.jnparray(bs))
            sigmas = gwnorm / matrix.jnp.sqrt(matrix.jnparray(bs))

            orfs = orf(matrix.jnparray(self.angles))

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return snr
        xs2snrs.cnt = cnt

        return xs2snrs

#    def sample_rhosigma(self, key, params, n=1, orf=hd_orfa):
    def sample_rhosigma(self, params, orf=hd_orfa):
        Phi = self.psls[0].gw.Phi.getN(params)
        sPhi = matrix.jnp.sqrt(Phi)

        Nmats, Fmats, Pmats, Tmats = zip(*[(psl.N.N.N, psl.N.F, psl.N.P_var.getN(params), psl.gw.F) for psl in self.psls])

        Ks = [matrix.WoodburyKernel_novar(matrix.NoiseMatrix1D_novar(Nmat), Fmat, matrix.NoiseMatrix1D_novar(Pmat))
              for Nmat, Fmat, Pmat in zip(Nmats, Fmats, Pmats)]
        K1s = [K.make_solve_1d() for K in Ks]

        TtKmTs = [Tmat.T @ K.solve_2d(Tmat)[0] for K, Tmat in zip(Ks, Tmats)]
        PsTtKmFsPs = [sPhi[:,matrix.jnp.newaxis] * (Tmat.T @ K.solve_2d(Fmat)[0]) * matrix.jnp.sqrt(Pmat)[matrix.jnp.newaxis,:]
                      for K, Tmat, Fmat, Pmat in zip(Ks, Tmats, Fmats, Pmats)]
        PsTts = [sPhi[:,matrix.jnp.newaxis] * Tmat.T for Tmat in Tmats]

        Ds = [sPhi[:,matrix.jnp.newaxis] * TtKmT * sPhi[matrix.jnp.newaxis,:] for TtKmT in TtKmTs]
        bs = [matrix.jnp.trace(Ds[i] @ Ds[j]) for (i,j) in self.pairs]

        cnt, iNs, iPs = 0, [], []
        for Nmat in Nmats:
            iNs.append(slice(cnt, cnt + Nmat.shape[0]))
            cnt += Nmat.shape[0]
        for Fmat in Fmats:
            iPs.append(slice(cnt, cnt + Fmat.shape[1]))
            cnt += Fmat.shape[1]

        def xs2snrs(xs):
            uks = [PsTt @ K1(matrix.jnp.sqrt(Nmat) * xs[iN])[0] + PsTtKmFsP @ xs[iP]
                   for PsTt, K1, Nmat, iN, PsTtKmFsP, iP in zip(PsTts, K1s, Nmats, iNs, PsTtKmFsPs, iPs)]

            # use with matrix.jnpnormal(key, cnt)
            ts = matrix.jnparray([matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in self.pairs])

            gwnorm = 10**(2.0 * params[self.gwpar])
            rhos = gwnorm * (matrix.jnparray(ts) / matrix.jnparray(bs))
            sigmas = gwnorm / matrix.jnp.sqrt(matrix.jnparray(bs))

            orfs = orf(matrix.jnparray(self.angles))

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return snr
        xs2snrs.cnt = cnt

        return xs2snrs

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
                U = matrix.jnp.linalg.cholesky(N, upper=True) # N = U^T U, so y = U^T x

                uks = [U @ k[0] for k in ks]
                ds = [U @ k[1] @ U.T for k in ks]

                ts = [matrix.jnp.dot(uks[i], uks[j].T) for (i,j) in pairs]
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

    def gx2cdf(self, params, osxs, cutoff=1e-6, limit=100, epsabs=1e-6):
        Qmat = self.Q(params)
        eigx = matrix.jnp.linalg.eigh(Qmat)[0]

        return eig2cdf(osxs, eigx, cutoff=cutoff, limit=limit, epsabs=epsabs)


@jax.jit
def imhof(u, x, eigs):
    theta = 0.5 * matrix.jnp.sum(matrix.jnp.arctan(eigs * u), axis=0) - 0.5 * x * u
    rho = matrix.jnp.prod((1.0 + (eigs * u)**2)**0.25, axis=0)

    return matrix.jnp.sin(theta) / (u * rho)

def eig2cdf(osxs, eigs, cutoff=1e-6, limit=100, epsabs=1e-6):
    # cutoff by number of eigenvalues is more friendly to jitted imhof
    eigs = eigs[:cutoff] if cutoff > 1 else eigs[matrix.jnp.abs(eigs) > cutoff]

    # jax.scipy.integrate is mostly not implemented. Could try quadax
    return np.array([0.5 - scipy.integrate.quad(lambda u: float(imhof(u, osx, eigs)),
                                                0, np.inf, limit=limit, epsabs=epsabs)[0] / np.pi for osx in osxs])
