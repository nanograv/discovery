import functools
# from dataclasses import dataclass

import numpy as np
import jax

from . import matrix
from . import signals

# import jax

# Kernel
#   ConstantKernel
#       define solve_1d, perhaps solve_2d (operate on numpy)
#   VariableKernel
#       define make_solve_1d, perhaps make_solve_2d (return functions that operate on jax)
#
#   all define make_kernelproduct (make_kernelterms)
#
# GP
#   ConstantGP
#       consists of a ConstantKernel and a numpy matrix
#   VariableGP
#       consists of a VariableKernel and a numpy matrix
#
# ShermanMorrisonKernel can return a ConstantKernel or a VariableKernel

# npta = je.PulsarLikelihood(je.residuals(psr),
#                            je.makenoise_measurement(psr, noisedict),
#                            je.makegp_ecorr(psr, noisedict),
#                            je.makegp_timing(psr),
#                            je.makegp_fourier('red_noise', psr, je.powerlaw, 10),
#                            je.makegp_fourier('red_noise', psr,
#                                              je.makepowerlaw_crn(5), 10, T=Tspan,
#                                              common=['crn_log10_A', 'crn_gamma']),
#                            concat=True)

class PulsarLikelihood:
    def __init__(self, args, concat=True):
        y     = [arg for arg in args if isinstance(arg, np.ndarray)]
        delay = [arg for arg in args if callable(arg)]
        noise = [arg for arg in args if isinstance(arg, matrix.Kernel)]
        cgps  = [arg for arg in args if isinstance(arg, matrix.ConstantGP)]
        vgps  = [arg for arg in args if isinstance(arg, matrix.VariableGP)]

        if len(y) > 1 or len(noise) > 1:
            raise ValueError("Only one residual vector and one noise Kernel allowed.")
        elif len(noise) == 0:
            raise ValueError("I need exactly one noise Kernel.")

        noise, y = noise[0], y[0]

        if cgps:
            if concat:
                cgp = matrix.CompoundGP(cgps)
                csm = matrix.ShermanMorrisonKernel(noise, cgp.F, cgp.Phi)
            else:
                csm = noise
                for cgp in cgps:
                    csm = matrix.ShermanMorrisonKernel(csm, cgp.F, cgp.Phi)
        else:
            csm = noise

        if vgps:
            if concat:
                vgp = matrix.CompoundGP(vgps)
                vsm = matrix.ShermanMorrisonKernel(csm, vgp.F, vgp.Phi)
            else:
                vsm = csm
                for vgp in vgps:
                    vsm = matrix.ShermanMorrisonKernel(vsm, vgp.F, vgp.Phi)
        else:
            vsm = csm

        if len(delay) > 0:
            delay = matrix.CompoundDelay(delay)

        self.y, self.delay, self.N = y, delay, vsm

        # a bit kludgy, we'll find a better way
        for gp in cgps + vgps:
            if hasattr(gp, 'name'):
                self.name = gp.name

    # allow replacement of residuals
    def __setattr__(self, name, value):
        if name == 'residuals' and 'logL' in self.__dict__:
            self.y = value
            del self.logL
        else:
            self.__dict__[name] = value

    @functools.cached_property
    def logL(self):
        if self.delay:
            return self.N.make_kernel(self.y, self.delay)
        else:
            return self.N.make_kernelproduct(self.y)

    @functools.cached_property
    def sample(self):
        return self.N.make_sample()


# for JAX purposes, it's best if os() returns a Python dictionary

# @dataclass
# class OS:
#     os: float
#     os_sigma: float
#     snr: float
#     rhos: np.array
#     sigmas: np.array


class GlobalLikelihood:
    def __init__(self, psls, globalgp=None):
        self.psls = psls
        self.globalgp = matrix.CompoundGlobalGP(globalgp) if isinstance(globalgp, list) else globalgp

    # allow replacement of residuals
    def __setattr__(self, name, value):
        if name == 'residuals':
            for psl, y in zip(self.psls, value):
                psl.y = y

            for p in ['os', 'os_rhosigma', 'logL', 'sample_conditional', 'conditional']:
                if p in self.__dict__:
                    delattr(self, p)
        else:
            self.__dict__[name] = value

    @functools.cached_property
    def sample(self):
        if self.globalgp is None:
            sls = [psl.sample for psl in self.psls]
            if len(sls) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            def sampler(key, params):
                ys = []
                for sl in sls:
                    key, y = sl(key, params)
                    ys.append(y)

                return key, ys

            sampler.params = sorted(set.union(*[set(sl.params) for sl in sls]))
        else:
            sls = [psl.sample for psl in self.psls]
            if len(sls) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            Phi_sample = self.globalgp.Phi.make_sample()

            Fs = [matrix.jnparray(F) for F in self.globalgp.Fs]

            i0, slcs = 0, []
            for F in self.globalgp.Fs:
                slcs.append(slice(i0, i0 + F.shape[1]))
                i0 = i0 + F.shape[1]

            def sampler(key, params):
                key, c = Phi_sample(key, params)

                ys = []
                for sl, F, slc in zip(sls, Fs, slcs):
                    key, y = sl(key, params)
                    ys.append(y + matrix.jnp.dot(F, c[slc]))

                # ys = [key, _ := sl(key, params) + jnp.dot(F, c[slc]) for sl, F, slc in zip(sls, Fs, slcs)]
                return key, ys

            sampler.params = sorted(set.union(*[set(sl.params) for sl in sls])) + Phi_sample.params

        return sampler

    @functools.cached_property
    def os(self):
        os_rhosigma = self.os_rhosigma

        gwpar = [par for par in self.globalgp.Phi.params if 'log10_A' in par][0]

        pos = self.globalgp.pos
        pairs = [(i1, i2) for i1 in range(len(pos)) for i2 in range(i1 + 1, len(pos))]
        # angles = [np.dot(pos[i1], pos[i2]) for (i1, i2) in pairs]
        orfs = matrix.jnparray([signals.hd_orf(pos[i1], pos[i2]) for (i1, i2) in pairs])

        def getos(params):
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10**(2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {'os': os, 'os_sigma': os_sigma, 'snr': snr, 'rhos': rhos, 'sigmas': sigmas,
                    'log10_A': params[gwpar]}

        getos.params = os_rhosigma.params
        # getos.angles = angles

        return getos

    @functools.cached_property
    def os_skyscramble(self):
        os_rhosigma = self.os_rhosigma

        gwpar = [par for par in self.globalgp.Phi.params if "log10_A" in par][0]

        pos = self.globalgp.pos
        pairs = [(i1, i2) for i1 in range(len(pos)) for i2 in range(i1 + 1, len(pos))]

        def hd_orf_os(pos1, pos2):
            """no need for autocorrelation if statement since
            this will only be used for the OS"""
            omc2 = (1.0 - matrix.jnp.dot(pos1, pos2)) / 2.0
            return 1.5 * omc2 * matrix.jnp.log(omc2) - 0.25 * omc2 + 0.5

        def getos(positions, params):
            orfs = matrix.jnparray([hd_orf_os(positions[i1], positions[i2]) for (i1, i2) in pairs])
            rhos, sigmas = os_rhosigma(params)

            gwnorm = 10 ** (2.0 * params[gwpar])
            rhos, sigmas = gwnorm * rhos, gwnorm * sigmas

            os = matrix.jnp.sum(rhos * orfs / sigmas**2) / matrix.jnp.sum(orfs**2 / sigmas**2)
            os_sigma = 1.0 / matrix.jnp.sqrt(matrix.jnp.sum(orfs**2 / sigmas**2))
            snr = os / os_sigma

            return {
                "os": os,
                "os_sigma": os_sigma,
                "snr": snr
            }

        getos.params = os_rhosigma.params
        # getos.angles = angles

        return getos

    # rho_ij = y_i Cmi(theta) Fi Phi Ftj Cmj(theta) y_j
    # sigma_ij = tr Fti Cmi(theta) Fi Phi Ftj Cmj(theta) Fj Phi
    @functools.cached_property
    def os_rhosigma(self):
        if self.globalgp is None:
            raise ValueError('GlobalLikelihood.os needs a globalgp.')
        else:
            pos = self.globalgp.pos
            pairs = [(i1, i2) for i1 in range(len(pos)) for i2 in range(i1 + 1, len(pos))]

            kernelsolves = [psl.N.make_kernelsolve(psl.y, Tmat) for (psl, Tmat) in zip(self.psls, self.globalgp.Fs)]

            getN = self.globalgp.Phi.getN
            components = self.globalgp.Fs[0].shape[1]

            def rhosigma(params):
                sN = matrix.jnp.sqrt(getN(params)[:components])
                ks = [k(params) for k in kernelsolves]

                ts = [matrix.jnp.dot(sN * ks[i][0], sN * ks[j][0]) for (i,j) in pairs]

                ds = [sN[:,matrix.jnp.newaxis] * k[1] * sN[matrix.jnp.newaxis,:] for k in ks]
                bs = [matrix.jnp.trace(ds[i] @ ds[j]) for (i,j) in pairs]

                return (matrix.jnp.array(ts) / matrix.jnp.array(bs),
                        1.0 / matrix.jnp.sqrt(matrix.jnp.array(bs)))

            rhosigma.params = sorted(set.union(*[set(k.params) for k in kernelsolves], getN.params))

            return rhosigma

    @functools.cached_property
    def logL(self):
        if self.globalgp is None:
            logls = [psl.logL for psl in self.psls]
            if len(logls) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            def loglike(params):
                return sum(logl(params) for logl in logls)

            loglike.params = sorted(set.union(*[set(logl.params) for logl in logls]))
        else:
            P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
            kterms = [psl.N.make_kernelterms(psl.y, Fmat) for psl, Fmat in zip(self.psls, self.globalgp.Fs)]
            if len(kterms) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            npsr = len(self.globalgp.Fs)
            ngp = self.globalgp.Fs[0].shape[1]

            def loglike(params):
                terms = [kterm(params) for kterm in kterms]

                p0 = sum([term[0] for term in terms])
                FtNmy = matrix.jnp.concatenate([term[1] for term in terms])

                Pinv, ldP = P_var_inv(params)

                #for i, term in enumerate(terms):
                #    Pinv = Pinv.at[i*ngp:(i+1)*ngp,i*ngp:(i+1)*ngp].add(term[2])
                #cf = matrix.jsp.linalg.cho_factor(Pinv)

                # this seems a bit slower than the .at/.set scheme in plogL below
                cf = matrix.jsp.linalg.cho_factor(Pinv + matrix.jsp.linalg.block_diag(*[term[2] for term in terms]))

                return p0 + 0.5 * (FtNmy.T @ matrix.jsp.linalg.cho_solve(cf, FtNmy) - ldP - 2.0 * matrix.jnp.sum(matrix.jnp.log(matrix.jnp.diag(cf[0]))))

            loglike.params = sorted(sorted(set.union(*[set(kterm.params) for kterm in kterms])) + P_var_inv.params)

        return loglike

    # MPI parallel likelihood
    @functools.cached_property
    def plogL(self):
        import mpi4py
        import mpi4jax

        mpicomm = mpi4py.MPI.COMM_WORLD
        jaxcomm = mpicomm.Clone()

        size = mpicomm.Get_size()
        rank = mpicomm.Get_rank()

        if self.globalgp is None:
            logls = [psl.logL for psl in self.psls]

            def loglike(params):
                slogl = sum(logl(params) for logl in logls)
                slogl, tk = mpi4jax.allreduce(slogl, mpi4py.MPI.SUM, comm=jaxcomm)
                return slogl

            local_list = sorted(set.union(*[set(logl.params) for logl in logls]))
            loglike.params = sorted(set([p for l in mpicomm.allgather(local_list) for p in l]))
        else:
            # handle the case where there are more matrices in self.globalgp than likelihoods
            Fmats = {name: Fmat for name, Fmat in zip(self.globalgp.name, self.globalgp.Fs)}
            kterms = [psl.N.make_kernelterms(psl.y, Fmats[psl.name]) for psl in self.psls]

            if rank == 0:
                npsr = len(self.globalgp.Fs)
                ngp = self.globalgp.Fs[0].shape[1]

                P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()

                def loglike(params):
                    b0 = matrix.jnp.zeros((size,), dtype=matrix.jnp.float64)
                    b1 = matrix.jnp.zeros((npsr, ngp), dtype=matrix.jnp.float64)
                    b2 = matrix.jnp.zeros((npsr, ngp, ngp), dtype=matrix.jnp.float64)

                    t0, t1, t2 = zip(*[kterm(params) for kterm in kterms])

                    b0 = b0.at[0].set(sum(t0))
                    b1 = b1.at[0::size,:].set(matrix.jnp.array(t1))
                    b2 = b2.at[0::size,:,:].set(matrix.jnp.array(t2))

                    for i in range(1, size):
                        b, tk = mpi4jax.recv(b0[i], source=i, tag=0, comm=jaxcomm)
                        b0 = b0.at[i].set(b)
                        b, tk = mpi4jax.recv(b1[i::size,:], source=i, tag=1, token=tk, comm=jaxcomm)
                        b1 = b1.at[i::size,:].set(b)
                        b, tk = mpi4jax.recv(b2[i::size,:,:], source=i, tag=2, token=tk, comm=jaxcomm)
                        b2 = b2.at[i::size,:,:].set(b)

                    p0 = matrix.jnp.sum(b0)
                    FtNmy = b1.flatten()

                    Pinv, ldP = P_var_inv(params)
                    cf = matrix.jsp.linalg.cho_factor(Pinv + matrix.jsp.linalg.block_diag(*b2))

                    ret = p0 + 0.5 * (FtNmy.T @ matrix.jsp.linalg.cho_solve(cf, FtNmy) - ldP - 2.0 * matrix.jnp.sum(matrix.jnp.log(matrix.jnp.diag(cf[0]))))
                    ret, tk = mpi4jax.bcast(ret, root=0, comm=jaxcomm)

                    return ret

                local_list = P_var_inv.params + sorted(set.union(*[set(kterm.params) for kterm in kterms]))
            else:
                def loglike(params):
                    t0, t1, t2 = zip(*[kterm(params) for kterm in kterms])

                    tk = mpi4jax.send(sum(t0), dest=0, tag=0, comm=jaxcomm)
                    tk = mpi4jax.send(matrix.jnp.array(t1), dest=0, tag=1, token=tk, comm=jaxcomm)
                    tk = mpi4jax.send(matrix.jnp.array(t2), dest=0, tag=2, token=tk, comm=jaxcomm)

                    ret, tk = mpi4jax.bcast(1.0, root=0, comm=jaxcomm)

                    return ret

                local_list = sorted(set.union(*[set(kterm.params) for kterm in kterms]))

            loglike.params = sorted(set([p for l in mpicomm.allgather(local_list) for p in l]))

        return loglike

    @functools.cached_property
    def sample_conditional(self):
        cond = self.conditional
        index = self.globalgp.index

        def sample_cond(key, params):
            mu, cf = cond(params)

            # conditional normal draws are obtained as `mu + y` after solving `cf.T y = x` for a normal deviate `x`
            key, subkey = matrix.jnpsplit(key)
            c = mu + matrix.jsp.linalg.solve_triangular(cf[0].T, matrix.jnpnormal(subkey, mu.shape), lower=False)

            return key, {par: c[sli] for par, sli in index.items()}

        sample_cond.params = cond.params

        return sample_cond

    @functools.cached_property
    def conditional(self):
        if self.globalgp is None:
            raise ValueError("Nothing to predict in GlobalLikelihood without a globalgp!")
        else:
            P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
            ndim = 1 if isinstance(self.globalgp.Phi, matrix.NoiseMatrix1D_var) else 2

            ksolves = [psl.N.make_kernelsolve(psl.y, Fmat) for psl, Fmat in zip(self.psls, self.globalgp.Fs)]

            if len(ksolves) == 0:
                raise ValueError('No PulsarLikelihoods in GlobalLikelihood: ' +
                    'if you provided them using a generator, it may have been consumed already. ' +
                    'In that case you can use a list.')

            if not ksolves[0].params:
                solves = [ksolve({}) for ksolve in ksolves]
                FtNmy = matrix.jnp.concatenate([solve[0] for solve in solves])

                # FtNmF = matrix.jsp.linalg.block_diag(*[solve[1] for solve in solves])
                FtNmFs = [solve[1] for solve in solves]
                ngp = FtNmFs[0].shape[0]

                def cond(params):
                    Pinv, _ = P_var_inv(params)

                    # Sm = (matrix.jnp.diag(Pinv) if ndim == 1 else Pinv) + FtNmF

                    Sm = matrix.jnp.diag(Pinv) if ndim == 1 else Pinv
                    for i, FtNmF in enumerate(FtNmFs):
                        Sm = Sm.at[i*ngp:(i+1)*ngp, i*ngp:(i+1)*ngp].add(FtNmF)

                    cf = matrix.jsp.linalg.cho_factor(Sm, lower=True)
                    mu = matrix.jsp.linalg.cho_solve(cf, FtNmy)

                    return mu, cf

                cond.params = P_var_inv.params
            else:
                def cond(params):
                    # each solve is a tuple TtSy, TtST
                    solves = [ksolve(params) for ksolve in ksolves]

                    FtNmy = matrix.jnp.concatenate([solve[0] for solve in solves])

                    Pinv, _ = P_var_inv(params)

                    # phiinv = (matrix.jnp.diag(Pinv) if Pinv.ndim == 1 else Pinv)
                    # tnt = matrix.jsp.linalg.block_diag(*[solve[1] for solve in solves])
                    # Sm = phiinv + tnt
                    Sm = (matrix.jnp.diag(Pinv) if ndim == 1 else Pinv) + matrix.jsp.linalg.block_diag(*[solve[1] for solve in solves])

                    # the variance of the normal is S = Sm^-1; but if we want normal deviates y
                    # with that variance, we can use the Cholesky decomposition
                    # S = L L^T => Sm = L^-T L^-1, and then solve L^-T y = x for randn x
                    # where cf = L^-1. See enterprise/signals/utils.py:ConditionalGP

                    # to get the actual covariance, one would use cho_solve(cf, identity matrix)

                    cf = matrix.jsp.linalg.cho_factor(Sm, lower=True)
                    mu = matrix.jsp.linalg.cho_solve(cf, FtNmy)

                    return mu, cf
                    # return mu, cf, phiinv, tnt

                cond.params = sorted(set.union(*[set(ksolve.params) for ksolve in ksolves])) + P_var_inv.params

        return cond


class ArrayLikelihood:
    def __init__(self, psls, commongp, globalgp=None, blockdiag='scipy'):
        self.psls = psls
        self.commongp = commongp # will want to combine if given a list
        self.globalgp = globalgp # will want to combine if given a list
        self.blockdiag = blockdiag

    @functools.cached_property
    def logL(self):
        Ns, ys = zip(*[(psl.N, psl.y) for psl in self.psls])
        vsm = matrix.VectorShermanMorrisonKernel_varP(Ns, self.commongp.F, self.commongp.Phi)

        if self.globalgp is None:
            loglike = vsm.make_kernelproduct(ys)
            loglike.params = self.commongp.Phi.params
        else:
            P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
            kterms = vsm.make_kernelterms(ys, self.globalgp.Fs)

            npsr = len(self.globalgp.Fs)
            ngp = self.globalgp.Fs[0].shape[1]
            blockdiag = self.blockdiag

            def loglike(params):
                terms = kterms(params)

                p0 = matrix.jnp.sum(terms[0])
                FtNmy = terms[1].reshape(npsr * ngp)

                Pinv, ldP = P_var_inv(params)

                # on CPU, all schemes have comparable compilation and performance times
                if blockdiag == 'scipy':
                    cf = matrix.jsp.linalg.cho_factor(Pinv + matrix.jsp.linalg.block_diag(*terms[2]))
                elif blockdiag == 'pyloop':
                    for i in range(npsr):
                        Pinv = Pinv.at[i*ngp:(i+1)*ngp,i*ngp:(i+1)*ngp].add(terms[2][i,:,:])
                    cf = matrix.jsp.linalg.cho_factor(Pinv)
                elif blockdiag == 'jaxloop':
                    Pinv = jax.lax.fori_loop(0, npsr,
                                lambda i, Pinv: jax.lax.dynamic_update_slice(Pinv,
                                    jax.lax.dynamic_slice(Pinv, (i*ngp,i*ngp), (ngp,ngp)) +
                                    jax.lax.squeeze(jax.lax.dynamic_slice(terms[2], (i,0,0), (1,ngp,ngp)), [0]),
                                    (i*ngp,i*ngp)),
                                Pinv)
                    cf = matrix.jsp.linalg.cho_factor(Pinv)
                return p0 + 0.5 * (FtNmy.T @ matrix.jsp.linalg.cho_solve(cf, FtNmy) - ldP - 2.0 * matrix.jnp.sum(matrix.jnp.log(matrix.jnp.diag(cf[0]))))

            loglike.params = sorted(kterms.params + P_var_inv.params)

        return loglike
