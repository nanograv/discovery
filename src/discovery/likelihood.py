import functools
# from dataclasses import dataclass

import numpy as np
import jax

from discovery import matrix
from discovery import signals

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
# WoodburyKernel can return a ConstantKernel or a VariableKernel

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
        # pgps  = [arg for arg in args if isinstance(arg, matrix.ComponentGP)]

        if len(y) > 1 or len(noise) > 1:
            raise ValueError("Only one residual vector and one noise Kernel allowed.")
        elif len(noise) == 0:
            raise ValueError("I need exactly one noise Kernel.")

        noise, y = noise[0], y[0]

        if cgps:
            if len(cgps) > 1 and concat:
                cgp = matrix.CompoundGP(cgps)
                csm = matrix.WoodburyKernel(noise, cgp.F, cgp.Phi)
            else:
                csm = noise
                for cgp in cgps:
                    csm = matrix.WoodburyKernel(csm, cgp.F, cgp.Phi)
        else:
            csm = noise

        if vgps:
            for vgp in vgps:
                if hasattr(vgp, 'gpname') and vgp.gpname == 'gw':
                    self.gw = vgp

            if len(vgps) > 1 and concat:
                vgp = matrix.CompoundGP(vgps)
                vsm = matrix.WoodburyKernel(csm, vgp.F, vgp.Phi)
                vsm.index = getattr(vgp, 'index', None)
                vsm.mean = getattr(vgp, 'mean', None)
            else:
                vsm = csm
                for vgp in vgps:
                    vsm = matrix.WoodburyKernel(vsm, vgp.F, vgp.Phi)
                    vsm.index = getattr(vgp, 'index', None)
                    vsm.mean = getattr(vgp, 'mean', None)
        else:
            vsm = csm

        if len(delay) > 0:
            y = matrix.CompoundDelay(y, delay)

        self.y, self.delay, self.N = y, delay, vsm

        # a bit kludgy, we'll find a better way
        for gp in cgps + vgps:
            if hasattr(gp, 'name'):
                self.name = gp.name

    # allow replacement of residuals
    def __setattr__(self, name, value):
        if name == 'residuals' and 'logL' in self.__dict__:
            self.y = value

            if len(self.delay) > 0:
                self.y = matrix.CompoundDelay(self.y, self.delay)

            del self.logL
        else:
            self.__dict__[name] = value

    @functools.cached_property
    def sample_conditional(self):
        cond = self.conditional
        index = self.N.index

        def sample_cond(key, params):
            mu, cf = cond(params)

            key, subkey = matrix.jnpsplit(key)
            c = mu + matrix.jsp.linalg.solve_triangular(cf[0].T, matrix.jnpnormal(subkey, mu.shape), lower=False)

            return key, {par: c[sli] for par, sli in index.items()}

        sample_cond.params = cond.params

        return sample_cond

    @functools.cached_property
    def conditional(self):
        if self.delay:
            raise NotImplementedError('No PulsarLikelihood.conditional with delays so far.')

        P_var_inv = self.N.P_var.Phi_inv or self.N.P_var.make_inv()

        ksolve = self.N.N.make_kernelsolve(self.y, self.N.F)

        if not ksolve.params:
            FtNmy, FtNmF = ksolve(params={})

            def cond(params):
                Pinv, _ = P_var_inv(params)
                Sm = (matrix.jnp.diag(Pinv) if Pinv.ndim == 1 else Pinv) + FtNmF
                cf = matrix.jsp.linalg.cho_factor(Sm, lower=True)
                mu = matrix.jsp.linalg.cho_solve(cf, FtNmy)

                return mu, cf

            cond.params = P_var_inv.params
        else:
            def cond(params):
                FtNmy, FtNmF = ksolve(params)
                Pinv, _ = P_var_inv(params)
                Sm = (matrix.jnp.diag(Pinv) if Pinv.ndim == 1 else Pinv) + FtNmF
                cf = matrix.jsp.linalg.cho_factor(Sm, lower=True)
                mu = matrix.jsp.linalg.cho_solve(cf, FtNmy)

                return mu, cf

            cond.params = sorted(ksolve.params) + P_var_inv.params

        return cond

    @functools.cached_property
    def clogL(self):
        if self.delay:
            raise NotImplementedError('No PulsarLikelihood.clogL with delays so far.')
        else:
            return self.N.make_kernelproduct_gpcomponent(self.y)

    @functools.cached_property
    def logL(self):
        return self.N.make_kernelproduct(self.y)

    @functools.cached_property
    def sample(self):
        if callable(self.y):
            raise NotImplementedError('No PulsarLikelihood.sample with delays so far.')
        else:
            return self.N.make_sample()


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

            # npsr = len(self.globalgp.Fs)
            # ngp = self.globalgp.Fs[0].shape[1]

            kmeans = getattr(self.globalgp, 'means', None)

            def loglike(params):
                terms = [kterm(params) for kterm in kterms]

                p0 = sum([term[0] for term in terms])
                FtNmy = matrix.jnp.concatenate([term[1] for term in terms])

                Pinv, ldP = P_var_inv(params)

                # for i, term in enumerate(terms):
                #     Pinv = Pinv.at[i*ngp:(i+1)*ngp,i*ngp:(i+1)*ngp].add(term[2])
                # cf = matrix.jsp.linalg.cho_factor(Pinv)

                # this seems a bit slower than the .at/.set scheme in plogL below
                FtNmF = matrix.jsp.linalg.block_diag(*[term[2] for term in terms])
                cf = matrix.jsp.linalg.cho_factor(Pinv + FtNmF)

                logp = p0 + 0.5 * (FtNmy.T @ matrix.jsp.linalg.cho_solve(cf, FtNmy) - ldP - 2.0 * matrix.jnp.sum(matrix.jnp.log(matrix.jnp.diag(cf[0]))))

                if kmeans is not None:
                    # -0.5 a0t.FtNmF.a0 + 0.5 a0t.FtNmF.Sm.FtNmF.a0 + a0t.FtNmy - a0t.FtNmF.Sm.FtNmy
                    # -0.5 (a0t.FtNmF).a0 + (FtNmy)t.a0 + 0.5 (a0t.FtNmF).Sm.FtNmF.a0 - (FtNmy)t.Sm.FtNmF.a0
                    # -0.5 (a0t.FtNmF).(a0 - Sm.FtNmF.a0) + (FtNmy)t.(a0 - Sm.FtNmF.a0)

                    a0 = kmeans(params)
                    FtNmFa0 = FtNmF @ a0
                    logp = logp - (0.5 * FtNmFa0.T - FtNmy.T) @ (a0 - matrix.jsp.linalg.cho_solve(cf, FtNmFa0))

                return logp

            params_kterms = list(set.union(*[set(kterm.params) for kterm in kterms]))
            params_kmeans = kmeans.params if kmeans is not None else []
            loglike.params = sorted(params_kterms + params_kmeans + P_var_inv.params)

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

                    Sm = matrix.jnp.diag(Pinv) if Pinv.ndim == 1 else Pinv
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
    def __init__(self, psls, *, commongp=None, globalgp=None, transform=None):
        self.psls = psls
        self.commongp = commongp
        self.globalgp = globalgp
        self.transform = transform

    # @functools.cached_property
    # def cloglast(self):
    #     commongp = matrix.VectorCompoundGP(self.commongp[:-1])
    #     lastgp = self.commongp[-1]

    #     Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])
    #     csm = matrix.VectorWoodburyKernel_varP(Ns, commongp.F, commongp.Phi)

    #     vsm = matrix.VectorWoodburyKernel_varP(Ns, lastgp.F, lastgp.Phi)
    #     if hasattr(lastgp, 'prior'):
    #         vsm.prior = lastgp.prior
    #     if hasattr(lastgp, 'index'):
    #         vsm.index = lastgp.index

    #     return vsm.make_kernelproduct_gpcomponent(self.ys)

    @functools.cached_property
    def clogL(self):
        if self.commongp is None and self.globalgp is None:
            def loglike(params):
                return sum(psl.clogL(params) for psl in self.psls)
            loglike.params = sorted(set.union(*[set(psl.clogL.params) for psl in self.psls]))

            return loglike
        elif self.commongp is None:
            # commongp = matrix.VectorCompoundGP(self.globalgp)
            raise NotImplementedError("ArrayLikelihood does not support a globalgp without a commongp")
        elif self.globalgp is None:
            # merge common GPs if necessary
            commongp = matrix.VectorCompoundGP(self.commongp)
        else:
            # merge common GPs and global GP
            cgp = self.commongp if isinstance(self.commongp, list) else [self.commongp]
            commongp = matrix.VectorCompoundGP(cgp + [self.globalgp])

        Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])
        self.vsm = matrix.VectorWoodburyKernel_varP(Ns, commongp.F, commongp.Phi)
        if hasattr(commongp, 'prior'):
            self.vsm.prior = commongp.prior
        if hasattr(commongp, 'index'):
            self.vsm.index = commongp.index

        loglike = self.vsm.make_kernelproduct_gpcomponent(self.ys, transform=self.transform)

        return loglike

    @functools.cached_property
    def logL(self):
        if self.commongp is None:
            if self.globalgp is None:
                def loglike(params):
                    return sum(psl.logL(params) for psl in self.psls)
                loglike.params = sorted(set.union(*[set(psl.logL.params) for psl in self.psls]))

                return loglike
            else:
                raise NotImplementedError("Currently ArrayLikelihood does not support a globalgp without a commongp")

        commongp = matrix.VectorCompoundGP(self.commongp)

        Ns, self.ys = zip(*[(psl.N, psl.y) for psl in self.psls])
        self.vsm = matrix.VectorWoodburyKernel_varP(Ns, commongp.F, commongp.Phi)
        self.vsm.index = getattr(commongp, 'index', None)

        if self.globalgp is None:
            loglike = self.vsm.make_kernelproduct(self.ys)
        else:
            P_var_inv = self.globalgp.Phi_inv or self.globalgp.Phi.make_inv()
            kterms = self.vsm.make_kernelterms(self.ys, self.globalgp.Fs)

            npsr = len(self.globalgp.Fs)
            ngp = self.globalgp.Fs[0].shape[1]

            kmeans = getattr(self.globalgp, 'means', None)

            def loglike(params):
                terms = kterms(params)

                p0 = matrix.jnp.sum(terms[0])
                FtNmy = terms[1].reshape(npsr * ngp)

                Pinv, ldP = P_var_inv(params)

                # alternatives to block_diag (with similar runtimes on CPU, slower on GPU)
                # for i in range(npsr):
                #    Pinv = Pinv.at[i*ngp:(i+1)*ngp,i*ngp:(i+1)*ngp].add(terms[2][i,:,:])
                #    cf = matrix.jsp.linalg.cho_factor(Pinv)
                #
                #    Pinv = jax.lax.fori_loop(0, npsr,
                #               lambda i, Pinv: jax.lax.dynamic_update_slice(Pinv,
                #                   jax.lax.dynamic_slice(Pinv, (i*ngp,i*ngp), (ngp,ngp)) +
                #                   jax.lax.squeeze(jax.lax.dynamic_slice(terms[2], (i,0,0), (1,ngp,ngp)), [0]),
                #                   (i*ngp,i*ngp)),
                #               Pinv)
                #    cf = matrix.jsp.linalg.cho_factor(Pinv)

                FtNmF = matrix.jsp.linalg.block_diag(*terms[2])
                cf = matrix.matrix_factor(Pinv + FtNmF)

                logp = p0 + 0.5 * (FtNmy.T @ matrix.matrix_solve(cf, FtNmy) - ldP - matrix.matrix_norm * matrix.jnp.sum(matrix.jnp.log(matrix.jnp.diag(cf[0]))))

                if kmeans is not None:
                    a0 = kmeans(params)
                    FtNmFa0 = FtNmF @ a0
                    logp = logp - (0.5 * FtNmFa0.T - FtNmy.T) @ (a0 - matrix.jsp.linalg.cho_solve(cf, FtNmFa0))

                return logp

            params_kmeans = kmeans.params if kmeans is not None else []
            loglike.params = sorted(kterms.params + params_kmeans + P_var_inv.params)

        return loglike
