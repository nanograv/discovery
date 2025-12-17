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
def noiseinv(graph, P):
    result = P.inv()


@mm.graph
def normal(g, y, Nsolve):
    Nmy, lN = Nsolve(y).split()
    logp = -0.5 * (y.T @ Nmy) - 0.5 * lN


@mm.graph
def woodbury(g, y, Nsolve, F, Pinv):
    Nmy, lN = Nsolve(y)
    FtNmy = F.T @ Nmy

    NmF, _ = Nsolve(F)
    FtNmF = F.T @ NmF

    Pm, lP = Pinv
    cf, lS = g.factor(Pm + FtNmF)

    logp = -0.5 * (y.T @ Nmy - FtNmy.T @ cf(FtNmy)) - 0.5 * (lN + lP + lS)


@mm.graph
def woodburysolve(g, y, Nsolve, F, Pinv):
    Nmy, lN = Nsolve(y)
    FtNmy = F.T @ Nmy

    NmF, _ = Nsolve(F)
    FtNmF = F.T @ NmF

    Pm, lP = Pinv
    cf, lS = g.factor(Pm + FtNmF)

    solve = Nmy - NmF @ cf(FtNmy)
    ld = lN + lP + lS

    result = g.pair(solve, ld)


@mm.graph
def globalwoodbury(g, ys, Nsolves, Fs, Pinv):
    ytNmys, FtNmys, FtNmFs = [], [], []

    if isinstance(Nsolves, (tuple, list)):
        for y, F, Nsolve in zip(ys, Fs, Nsolves):
            Nmy, lN = Nsolve(y)
            NmF, _ = Nsolve(F)

            ytNmys.append(y.T @ Nmy + lN)
            FtNmys.append(F.T @ Nmy)
            FtNmFs.append(F.T @ NmF)
    else:
        Nmy_lNs = Nsolves(*ys)
        NmF_lNs = Nsolves(*Fs)

        for i, (y, F) in enumerate(zip(ys, Fs)):
            Nmy, lN = Nmy_lNs[i].split()
            NmF, _  = NmF_lNs[i].split()

            ytNmys.append(y.T @ Nmy + lN)
            FtNmys.append(F.T @ Nmy)
            FtNmFs.append(F.T @ NmF)

    ytNmy = g.node(lambda *fts: sum(fts), inputs=ytNmys)
    FtNmy = g.node(lambda *vecs: jnp.hstack(vecs), inputs=FtNmys)
    FtNmF = g.node(lambda *mats: jsp.linalg.block_diag(*mats), inputs=FtNmFs)

    Pm, lP = Pinv
    cf, lS = g.factor(Pm + FtNmF)

    lP = lP.sum() # g.node(lambda x: jnp.sum(x), inputs=[lP]) # should be an op

    logp = -0.5 * (ytNmy - g.dot(FtNmy, cf(FtNmy))) - 0.5 * (lP + lS)


@mm.graph
def vectorwoodbury(g, ys, Nsolves, Fs, Pinv):
    ytNmys, FtNmys, FtNmFs = [], [], []

    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)

        ytNmys.append(y.T @ Nmy + lN)
        FtNmys.append(F.T @ Nmy)
        FtNmFs.append(F.T @ NmF)

    ytNmy = g.node(lambda *fts: sum(fts), inputs=ytNmys)
    FtNmy = g.node(lambda *vecs: jnp.array(vecs), inputs=FtNmys)
    FtNmF = g.node(lambda *mats: jnp.array(mats), inputs=FtNmFs)

    Pm, lP = Pinv
    cf, lS = g.factor(Pm + FtNmF)

    lP = lP.sum()
    lS = lS.sum()

    logp = -0.5 * (ytNmy - g.dot(FtNmy, cf(FtNmy))) - 0.5 * (lP + lS)


@mm.graph
def vectorwoodburysolve(g, ys, Nsolves, Fs, Pinv):
    Nmys, NmFs, FtNmys, FtNmFs, lNs = [], [], [], [], []

    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y) # Nmy: (n_i,), lN: scalar
        NmF, _ = Nsolve(F)  # NmF: (n_i, k)

        lNs.append(lN)
        Nmys.append(Nmy); NmFs.append(NmF)
        FtNmys.append(F.T @ Nmy); FtNmFs.append(F.T @ NmF) # F.T @ Nmy: (k,); F.T @ NmF: (k, k)

    FtNmy = g.node(lambda *vecs: jnp.array(vecs), inputs=FtNmys) # FtNmy: (np, k)
    FtNmF = g.node(lambda *mats: jnp.array(mats), inputs=FtNmFs) # FtNmF: (np, k, k)

    Pm, lP = Pinv                  # Pm: (np, k, k), lP: (np,)
    cf, lS = g.factor(Pm + FtNmF)  # why is lS just one number???

    cfFtNmy = cf(FtNmy)            # cfFtNmy: (np, k)

    solves = []
    for i, (Nmy, NmF) in enumerate(zip(Nmys, NmFs)):
       solves.append(g.pair(Nmy - NmF @ cfFtNmy[i, :], lNs[i] + lP[i] + lS[i]))

    result = g.node(lambda *ss: tuple(ss), inputs=solves)



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
        return mm.func(noisesolve(None, self.N))

    @property
    def make_inv(self):
        return mm.func(noiseinv(self.N))

    def make_kernelproduct(self, y):
        return mm.func(normal(y, self.make_solve))


class WoodburyKernel(matrix.Kernel):
    def __init__(self, N, F, P):
        self.N, self.F, self.P = N, F, P

    @property
    def make_solve(self):
        return mm.func(woodburysolve(None, self.N.make_solve, self.F, self.P.make_inv))

    def make_kernelproduct(self, y):
        return mm.func(woodbury(y, self.N.make_solve, self.F, self.P.make_inv))


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


class CompoundGP:
    def __new__(cls, x):
        if not isinstance(x, (list, tuple)):
            # if this is a single GP, just return it
            return x
        else:
            return super().__new__(cls)

    def __init__(self, gplist):
        self.gplist = gplist

    def _concat(self, vecmats):
        return functools.reduce(lambda x, y: mm.func(concat(x, y)), vecmats)

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


# monkeypatching

signals.matrix.NoiseMatrix1D_novar = NoiseMatrix
signals.matrix.NoiseMatrix1D_var = NoiseMatrix
signals.matrix.CompoundGP = CompoundGP
signals.matrix.CompoundDelay = CompoundDelay
