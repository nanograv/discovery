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
            FtNmys.append(g.dot(F, Nmy))
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
    logp = -0.5 * (ytNmy - g.dot(FtNmy, mu)) - 0.5 * (lP.sum() + lS.sum())


# this is different enough that it's OK to have a separate graph for the solve
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
            return mm.func(globalwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv))
        else:
            return mm.func(globalwoodbury(ys, self.Ns.make_solve, self.Fs, self.P.make_inv))


class VectorWoodburyKernel(matrix.Kernel):
    def __init__(self, Ns, Fs, P):
        self.Ns, self.Fs, self.P = Ns, Fs, P

    @property
    def make_solve(self):
        return mm.func(vectorwoodburysolve([None] * len(self.Ns), [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv))

    def make_kernelproduct(self, ys):
        return mm.func(vectorwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv))

    def make_conditional(self, ys):
        return mm.func(vectorwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv), output='cond')


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
        # TO DO: won't work for 2D priors
        N = self._concat([gp.Phi.N for gp in self.gplist])
        return NoiseMatrix(N)


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
