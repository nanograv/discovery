import functools

import jax.numpy as jnp
import jax.scipy as jsp

from . import matrix
from . import signals
from . import metamatrix as mm



@mm.graph
def noisesolve(graph, y, N):
    result = N.solve(y)


@mm.graph
def noiseinvfunc(graph, P):
    result = P


@mm.graph
def noiseinv(graph, P):
    result = P.inv()


@mm.graph
def normal(g, y, Nsolve):
    Nmy, lN = Nsolve(y).split()
    logp = -0.5 * (y.T @ Nmy) - 0.5 * lN


# this is actually pretty inefficient on CPU
stacksolve = False

@mm.graph
def woodbury(g, y, Nsolve, F, Pinv):
    if stacksolve:
        (Nmy, NmF), lN = g.stacksolve(Nsolve, y, F)
    else:
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)

    FtNmy = g.dot(NmF, y) # FtNmy = g.dot(F, Nmy)
    FtNmF = g.dot(F, NmF)

    Pm, lP = Pinv # should be a call even without parameters
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)
    ld = g.node(lambda lN, lP, lS: lN + lP + lS, [lN, lP, lS], description=f'{lN.name} + {lP.name} + {lS.name}')

    cond = g.pair(mu, cf, name='cond')
    solve = g.pair(Nmy - NmF @ mu, ld, name='solve')

    logp = -0.5 * (g.dot(y, Nmy) - g.dot(FtNmy, mu)) - 0.5 * ld

    # more readable, but doing this keeps y.T @ Nmy from being cached
    # logp = g.node(lambda y, Nmy, FtNmy, mu, ld: -0.5 * (y.T @ Nmy - FtNmy.T @ mu) - 0.5 * ld,
    #               [y, Nmy, FtNmy, mu, ld], description=f'-0.5 * (y.T @ Nmy - FtNmy.T @ mu) - 0.5 * ld')


@mm.graph
def woodburylatent(g, y, Nsolve, F, Psolve, getc):
    c = getc

    yp = y - F @ c
    Nmyp, lN = Nsolve(yp)

    Pmc, lP = Psolve(c)

    logp = -0.5 * (yp.T @ Nmyp + c.T @ Pmc + lP + lN)


@mm.graph
def globalwoodbury(g, ys, Nsolves, Fs, Pinv):
    ytNmys, FtNmys, FtNmFs = [], [], []

    if isinstance(Nsolves, (tuple, list)):
        for y, F, Nsolve in zip(ys, Fs, Nsolves):
            Nmy, lN = Nsolve(y)
            NmF, _ = Nsolve(F)

            ytNmys.append(g.dot(y, Nmy) + lN)
            FtNmys.append(g.dot(NmF, y))
            # FtNmys.append(g.dot(F, Nmy))
            FtNmFs.append(g.dot(F, NmF))
    else:
        # ingest output from vectorwoodburysolve
        Nmy_lNs = Nsolves(*ys)
        NmF_lNs = Nsolves(*Fs)

        # we can't include Nmy_lNs/NmF_lNs in the zip
        # since it's a tuple Sym that doesn't know its length
        for i, (y, F) in enumerate(zip(ys, Fs)):
            Nmy, lN = Nmy_lNs[i].split()
            NmF, _  = NmF_lNs[i].split()

            ytNmys.append(g.dot(y, Nmy) + lN)
            FtNmys.append(g.dot(F, Nmy))
            FtNmFs.append(g.dot(F, NmF))

    ytNmy = g.sum_all(ytNmys)
    FtNmy = g.hstack(FtNmys)
    FtNmF = g.block_diag(FtNmFs)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)

    lP = lP.sum() # g.node(lambda x: jnp.sum(x), inputs=[lP]) # should be an op

    logp = -0.5 * (ytNmy - g.dot(FtNmy, g.cho_solve(cf, FtNmy))) - 0.5 * (lP + lS)

@mm.graph
def globalwoodbury_fused(g, projected, Pinv):
    ytNmy_proj, ld, FtNmy_proj, FtNmF_proj = (projected[0], projected[1],
                                                projected[2], projected[3])
    ytNmy = g.sum(ytNmy_proj)
    total_ld = g.sum(ld)
    FtNmy = g.node(lambda x: x.reshape(-1), [FtNmy_proj])         # (67*28,)
    FtNmF = g.node(lambda x: jsp.linalg.block_diag(*x), [FtNmF_proj])  # (1876, 1876)
    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    logp = -0.5 * (ytNmy - g.dot(FtNmy, g.cho_solve(cf, FtNmy))) - 0.5 * (lP.sum() + total_ld + lS)


@mm.graph
def vectorwoodburyjointsolve(g, ys, Fs_outer, Nsolves, Fs_inner, Pinv):
    """Jointly solve — provides both TOA-space and projected outputs."""
    Nmys, NmFs_out, NmFs_in = [], [], []
    FtNmys_in, FtNmFs_in = [], []
    lNs = []

    for y, F_out, F_in, Nsolve in zip(ys, Fs_outer, Fs_inner, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF_out, _ = Nsolve(F_out)
        NmF_in, _ = Nsolve(F_in)
        lNs.append(lN)
        Nmys.append(Nmy)
        NmFs_out.append(NmF_out)
        NmFs_in.append(NmF_in)
        FtNmys_in.append(g.dot(F_in, Nmy))
        FtNmFs_in.append(g.dot(F_in, NmF_in))

    FtNmy_in = g.array(FtNmys_in)
    FtNmF_in = g.array(FtNmFs_in)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF_in)
    mu_y = g.cho_solve(cf, FtNmy_in)

    FtNmFs_cross = [g.dot(F_in, NmF_out) for F_in, NmF_out in zip(Fs_inner, NmFs_out)]
    FtNmF_cross = g.array(FtNmFs_cross)
    mu_F = g.cho_solve(cf, FtNmF_cross)

    # --- TOA-space output (for make_solve etc.) ---
    solves = []
    for i in range(len(ys)):
        ld = lNs[i] + lP[i] + lS[i]
        y_corr = Nmys[i] - NmFs_in[i] @ mu_y[i, :]
        F_corr = NmFs_out[i] - NmFs_in[i] @ mu_F[i, :, :]
        solves.append(g.pair(g.pair(y_corr, ld), g.pair(F_corr, ld)))
    # if you just want the results normally you can prune to here
    g.named(g.ntuple(solves), 'result')

    # --- Projected output (for globalwoodbury_fused) ---
    ytNmy_consts = g.array([g.dot(ys[i], Nmys[i]) for i in range(len(ys))])          # (67,)
    FtNmy_out = g.array([g.dot(Fs_outer[i], Nmys[i]) for i in range(len(ys))])       # (67, 28)
    FtNmF_cross_out = g.array([g.dot(Fs_outer[i], NmFs_in[i]) for i in range(len(ys))])  # (67, 28, 60)
    FtNmF_out = g.array([g.dot(Fs_outer[i], NmFs_out[i]) for i in range(len(ys))])   # (67, 28, 28)
    ld = g.array(lNs) + lP + lS                                                       # (67,)

    # Batched runtime: ~5 nodes instead of ~2000
    ytNmy_proj = ytNmy_consts - g.node(lambda a, b: jnp.einsum('ij,ij->i', a, b),
                                        [FtNmy_in, mu_y])                              # (67,)
    FtNmy_proj = FtNmy_out - g.node(lambda A, x: jnp.einsum('ijk,ik->ij', A, x),
                                     [FtNmF_cross_out, mu_y])                          # (67, 28)
    FtNmF_proj = FtNmF_out - FtNmF_cross_out @ mu_F                                   # (67, 28, 60) @ (67, 60, 28) = (67, 28, 28)

    g.named(g.ntuple([ytNmy_proj, ld, FtNmy_proj, FtNmF_proj]), 'projected')

@mm.graph
def vectorwoodbury(g, ys, Nsolves, Fs, Pinv):
    ytNmys, FtNmys, FtNmFs = [], [], []

    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)

        ytNmys.append(y.T @ Nmy + lN)
        FtNmys.append(F.T @ Nmy)
        FtNmFs.append(F.T @ NmF)

    ytNmy = g.sum_all(ytNmys)
    FtNmy = g.array(FtNmys)
    FtNmF = g.array(FtNmFs)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)

    cond = g.pair(mu, cf, name='cond')
    logp = -0.5 * (ytNmy - g.sum(FtNmy * mu)) - 0.5 * (lP.sum() + lS.sum())


@mm.graph
def vectorwoodburysolve(g, ys, Nsolves, Fs, Pinv):
    Nmys, NmFs, FtNmys, FtNmFs, lNs = [], [], [], [], []

    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y) # Nmy: (n_i,), lN: scalar
        NmF, _ = Nsolve(F)  # NmF: (n_i, k)

        lNs.append(lN)
        Nmys.append(Nmy)
        NmFs.append(NmF)
        FtNmys.append(g.dot(F, Nmy))
        FtNmFs.append(g.dot(F, NmF)) # F.T @ Nmy: (k,); F.T @ NmF: (k, k)

    FtNmy = g.array(FtNmys)            # FtNmy: (np, k)
    FtNmF = g.array(FtNmFs)            # FtNmF: (np, k, k)

    Pm, lP = Pinv                      # Pm: (np, k, k), lP: (np,)
    cf, lS = g.cho_factor(Pm + FtNmF)  # cf: (np, k, k), lS: (np,)
    mu = g.cho_solve(cf, FtNmy)        # cfFtNmy: (np, k)

    solves = []
    for i, (Nmy, NmF) in enumerate(zip(Nmys, NmFs)):
       solves.append(g.pair(Nmy - NmF @ mu[i, :], lNs[i] + lP[i] + lS[i]))

    solve = g.ntuple(solves)



@mm.graph
def concat(g, a, b):
    if isinstance(a, (list, tuple)):
        result = [g.node(lambda x, y: jnp.hstack([x, y]), [ai, bi]) for ai, bi in zip(a, b)]
    else:
        result = g.node(lambda x, y: jnp.hstack([x, y]), [a, b])

@mm.graph
def delay(g, y, d):
    result = y - d



class NoiseMatrix(matrix.Kernel):
    def __init__(self, N):
        self.N = N

    @property
    def make_solve(self):
        return noisesolve(None, self.N)

    @property
    def make_inv(self):
        if getattr(self, 'inv', None):
            # shortcut for GPs with custom make_inv
            return noiseinvfunc(self.inv)
        else:
            return noiseinv(self.N)

    def make_kernelproduct(self, y):
        return normal(y, self.make_solve)


class WoodburyKernel(matrix.Kernel):
    def __init__(self, N, F, P):
        self.N, self.F, self.P = N, F, P

    @property
    def make_solve(self):
        return mm.prune_graph(woodbury(None, self.N.make_solve, self.F, self.P.make_inv), output='solve')

    def make_kernelproduct(self, y):
        return woodbury(y, self.N.make_solve, self.F, self.P.make_inv)

    def make_conditional(self, y):
        return mm.prune_graph(woodbury(y, self.N.make_solve, self.F, self.P.make_inv), output='cond')

    def make_coefficientproduct(self, y):
        cvars = list(self.index.keys())
        def getc(params):
            return jnp.concatenate([params[cvar] for cvar in cvars])
        getc.params = cvars

        return woodburylatent(y, self.N.make_solve, self.F, self.P.make_solve, getc)


class GlobalWoodburyKernel(matrix.Kernel):
    def __init__(self, Ns, Fs, P):
        self.Ns, self.Fs, self.P = Ns, Fs, P

    def make_kernelproduct(self, ys):
        if isinstance(self.Ns, (tuple, list)):
            # old path: list of per-pulsar noise kernels
            return globalwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv)
        elif hasattr(self.Ns, 'Ns'):
            # compound kernel: vectorized path
            joint_graph = vectorwoodburyjointsolve(
                ys, self.Fs, [N.make_solve for N in self.Ns.Ns],
                self.Ns.Fs, self.Ns.P.make_inv)
            proj_graph = mm.prune_graph(joint_graph, output='projected')
            return globalwoodbury_fused(proj_graph, self.P.make_inv)
        else:
            # single noise matrix
            return globalwoodbury(ys, self.Ns.make_solve, self.Fs, self.P.make_inv)


class VectorWoodburyKernel(matrix.Kernel):
    def __init__(self, Ns, Fs, P):
        self.Ns, self.Fs, self.P = Ns, Fs, P

    @property
    def make_solve(self):
        return vectorwoodburysolve([None] * len(self.Ns),
                                   [N.make_solve for N in self.Ns],
                                   self.Fs, self.P.make_inv)

    def make_kernelproduct(self, ys):
        return vectorwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv)

    def make_conditional(self, ys):
        return mm.prune_graph(vectorwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv), output='cond')

    def make_joint_solve(self, Fs_outer):
        return vectorwoodburyjointsolve(
            [None] * len(self.Ns),   # ys are args
            Fs_outer,                 # global GP bases, constants
            [N.make_solve for N in self.Ns],
            self.Fs,
            self.P.make_inv
        )


class CompoundGP:
    def __new__(cls, x):
        if not isinstance(x, (list, tuple)):
            # if this is a single GP, just return it
            return x
        else:
            return super().__new__(cls)
    def __init__(self, gplist):
        self.gplist = gplist
        if all(hasattr(gp, 'index') for gp in gplist):
            self.index = {k: v for d in gplist for k, v in d.index.items()}
    def _concat(self, vecmats):
        return functools.reduce(lambda x, y: concat(x, y), vecmats)
    @property
    def F(self):
        # for VectorWoodburyKernel
        if all(isinstance(gp.F, tuple) for gp in self.gplist):
            return tuple(self._concat(Fs) for Fs in zip(*(gp.F for gp in self.gplist)))
        else:
            return self._concat([gp.F for gp in self.gplist])

    @property
    def Phi(self):
        N = self._concat([gp.Phi.N for gp in self.gplist])
        nm = NoiseMatrix(N)

        if any(hasattr(gp, 'Phi_inv') for gp in self.gplist):
            print('hi')
            phi_invs = [gp.Phi_inv for gp in self.gplist]

            def combined_inv(params):
                results = [f(params) for f in phi_invs]
                precisions = [r[0] for r in results]
                logdets = [r[1] for r in results]
                return (jax.scipy.linalg.block_diag(*precisions), sum(logdets))

            combined_inv.args = list(dict.fromkeys(
                arg for f in phi_invs for arg in getattr(f, 'args', [])
            ))
            combined_inv.params = list(dict.fromkeys(
                p for f in phi_invs for p in getattr(f, 'params', [])
            ))
            nm.inv = combined_inv

        return nm

def CompoundDelay(residuals, delays):
    return functools.reduce(lambda x, y: mm.func(delay(x, y)), [residuals, *delays])


### experimental

@mm.graph
def woodburyfast(g, y, allsolve, F, Pinv):
    ytNmy, FtNmy, FtNmF, lN = allsolve(y, F)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)
    ld = lP + lS + lN

    # cond = g.pair(mu, cf, name='cond')
    # solve = g.pair(Nmy - NmF @ mu, ld, name='solve')

    logp = -0.5 * (ytNmy - g.dot(FtNmy, mu)) - 0.5 * ld

# yt Km y = yt Nm y - yt Nm F (Pinv + FtNmF)^-1 Ft Nm y
# Tt Km y = Tt Nm y - Tt Nm F (Pinv + FtNmF)^-1 Ft Nm y
# Tt Km T = Tt Nm T - Tt Nm F (Pinv + FtNmF)^-1 Ft Nm T
# quindi mi mancano TtNmy, TtNmF, TtNmT;
# il primo e l'ultimo si possono ottenere da allsolve(y, T), ma TtNmF?

@mm.graph
def noiseallsolve(graph, y, F, N):
    result = N.allsolve(y, F)
