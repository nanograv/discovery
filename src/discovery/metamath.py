"""metamath: graph-based kernel/GP classes built on the metamatrix DSL.

Kernel-math methods in this file return metamatrix **graphs**, not closures.
Sub-objects (`N.make_solve`, `P.make_inv`, reparams, ExtSignal coeff maps,
priors, ...) compose into those graphs as leaves so `fold_constants` can
decide what runs at trace time vs runtime. `mm.func` belongs at the outer
boundary in `likelihood.py`, not inside methods here. `make_sample` is the
one documented exception.

See `docs/metamatrix_dev.md` and `docs/design/metamatrix/architecture.md`
for conventions and porting guidance before adding new methods.
"""

import functools

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from . import signals
from . import metamatrix as mm
from .utils import (
    Kernel,
    make_uind,
    smup_ind_correct,
    vsmup_ind_correct,
    vsmdp_ind,
    smwhiten_ind_correct,
    vsmwhiten_ind_correct,
)



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

    # Single-precision (stage 2): pin the white-noise logdet lN and the prior
    # logdet lP to float64. lS (the small Phi^-1 + FtNmF Cholesky logdet) is left
    # in the working dtype on purpose -- pinning it would force that Cholesky into
    # float64, which is the cost we are avoiding. ld stays working dtype: it is the
    # final sum, and protecting that sum is Piece 2's job, not the pins'.
    g.pin_f64(lN)
    g.pin_f64(lP)
    ld = g.node(lambda lN, lP, lS: lN + lP + lS, [lN, lP, lS], description=f'{lN.name} + {lP.name} + {lS.name}')

    cond = g.pair(mu, cf, name='cond')
    solve = g.pair(Nmy - NmF @ mu, ld, name='solve')

    # ytNmy = y^T N^-1 y is the catastrophic-cancellation term: pin it to float64
    # so its ntoa-long accumulation is built accurately even under a float32
    # working dtype.
    ytNmy = g.pin_f64(g.dot(y, Nmy))
    # Half A: combine the final logL in float64 -- ytNmy/lN/lP are read in as f64
    # (no loss), FtNmy.mu and lS are cast up -- so the O(1) result is not rounded
    # to the float32 ulp of a ~1e6 term. The float32 Cholesky / mu upstream are
    # untouched (that accuracy is Half B / reference+delta).
    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, mu), [lN, lP, lS])

    # more readable, but doing this keeps y.T @ Nmy from being cached
    # logp = g.node(lambda y, Nmy, FtNmy, mu, ld: -0.5 * (y.T @ Nmy - FtNmy.T @ mu) - 0.5 * ld,
    #               [y, Nmy, FtNmy, mu, ld], description=f'-0.5 * (y.T @ Nmy - FtNmy.T @ mu) - 0.5 * ld')


@mm.graph
def woodbury_proj(g, y, Nwhiten, M, F, Pinv):
    """Marginal logL with the timing model M handled by *projection* instead of
    a huge-variance prior (the float32-safe path; see
    docs/design/single_precision/ (timing-model projection).

    Same marginal likelihood as ``woodbury`` built from the combined basis
    ``[M | F]`` with a flat (sigma^2 -> infinity) prior on the M coefficients,
    but the flat-prior limit is taken analytically: we whiten by K = N + ECORR,
    then orthogonally project the whitened data and Fourier basis out of the
    span of the whitened timing basis. Nothing ever forms the 1e40 prior block
    that overflows float32.

    Inputs
      y       : residuals (free leaf, applied later).
      Nwhiten : `N.make_whiten` GraphLeaf, applies W = K^{-1/2} -> (W x, logdet K).
      M       : timing-model design matrix (n_toa, m_tm) -- projected OUT.
      F       : Fourier GP basis (n_toa, k)             -- kept.
      Pinv    : GP prior inverse leaf -> (Phi^-1, logdet Phi) (GP block only,
                NO timing block).

    Whiten FIRST: forming M^T K^-1 M by un-whitened sums
    re-introduces large-minus-large cancellation in the low-frequency Fourier
    modes that overlap the timing polynomials -- catastrophic in float32.
    """
    yw, lN = Nwhiten(y).split()      # W y, logdet K  (lN used once)
    A, _ = Nwhiten(M).split()        # W M  (whitened timing basis)
    B, _ = Nwhiten(F).split()        # W F  (whitened Fourier basis)

    # One factorization of the (well-conditioned) timing Gram A^T A = M^T K^-1 M.
    # cho_factor hands back its logdet for free -> that IS the timing Jacobian
    # logdet(M^T K^-1 M); no separate SVD-logdet precompute needed.
    AtA = g.dot(A, A)
    cfA, ldA = g.cho_factor(AtA)

    # Orthogonally project y and F out of the (whitened) timing span:
    #   r_perp = W y - W M (A^T A)^-1 A^T (W y)   = W (y - M eps_hat)   (GLS resid)
    #   B_perp = W F - W M (A^T A)^-1 A^T (W F)
    coeffy = g.cho_solve(cfA, g.dot(A, yw)); r_perp = yw - A @ coeffy
    coeffB = g.cho_solve(cfA, g.dot(A, B));  B_perp = B - A @ coeffB

    FtNmF = g.dot(B_perp, B_perp)    # F_perp^T K^-1 F_perp
    FtNmy = g.dot(B_perp, r_perp)    # F_perp^T K^-1 r_perp
    # ytNmy = r_perp^T K^-1 r_perp -- the catastrophic-cancellation term, pin f64.
    ytNmy = g.pin_f64(g.dot(r_perp, r_perp))

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)

    # Mirror woodbury's pins: white-noise logdet lN and prior logdet lP to f64;
    # the timing Jacobian ldA and the small-matrix logdet lS stay working dtype.
    g.pin_f64(lN)
    g.pin_f64(lP)
    ld = g.node(lambda a, b, c, d: a + b + c + d, [lN, ldA, lP, lS],
                description='logdetK + logdet(MtKmM) + lP + lS')

    # TOA-space output (for make_solve / the fused array path): apply the
    # timing-projected inverse operator A = W (P_perp - P_perp B (Pinv+B^T B)^-1 B^T P_perp) W
    # to the input. r_perp = P_perp(Wy), B_perp = P_perp(WF); the kept-GP Woodbury
    # correction is `B_perp @ mu`, and the trailing whitening W maps back to TOA
    # space. Applied to the outer commongp/globalgp bases by the fused kernel,
    # this propagates the timing projection through the whole array for free.
    solve_w, _ = Nwhiten(r_perp - B_perp @ mu).split()
    solve = g.pair(solve_w, ld, name='solve')

    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, mu), [lN, ldA, lP, lS])


@mm.graph
def woodbury_refdelta(g, y, Nsolve, F, Pinv, Pinv_ref):
    """Marginal logL as a *reference + delta* expansion (Piece 2 'Half B'; the
    single-level test rung). See
    docs/design/single_precision/ (reference+delta)
    (sec. 2-3) and docs/adr/0001,0003.

    Instead of computing logL(theta) directly -- a catastrophic float32
    cancellation `ytNmy - FtNmy.mu` of two ~1e6 numbers giving an O(1) result --
    we write

        logL(theta) = logL_ref + Delta logL,        Delta logL = 1/2 dQ - 1/2 dLdet,

    with logL_ref computed ONCE in float64 at a frozen reference covariance
    Phi_ref, and only the small O(1) increment per call. `ytNmy`, logdet N, and
    n*log2pi cancel exactly in the difference (fixed white noise), so they never
    enter Delta logL. The increment lives entirely in the small n_gp space and is
    formed WITHOUT subtracting two large numbers, so float32 holds it to ~f32-eps
    of an O(1) value (~1e-7) rather than the ~1e-2 floor of the direct path.

    Inputs mirror ``woodbury`` plus one frozen reference:
      Pinv     : current  Phi^-1 leaf -> (Phi^-1, logdet Phi)        (live).
      Pinv_ref : reference Phi_ref^-1 leaf (constant -> folds to f64).

    Math (companion note):
      v = F^T N^-1 y,  G = F^T N^-1 F   (both fixed under fixed WN);
      C = Phi^-1 + G,  u = C^-1 v,  w = Phi^-1 u   (and _ref at Phi_ref);
      dPhi = Phi - Phi_ref               (benign covariance-space increment);
      dQ    = w^T dPhi w_ref             (sec. 2, f64-accumulated);
      dLdet = slogdet(I + S0^-1 dPhi G), S0 = I + Phi_ref G   (sec. 3).
    """
    Nmy, lN = Nsolve(y)
    NmF, _ = Nsolve(F)
    v = g.dot(NmF, y)            # FtNmy  (fixed)
    G = g.dot(F, NmF)            # FtNmF  (fixed)
    # ytNmy cancels in Delta logL but is needed once to build logL_ref; pin f64.
    ytNmy = g.pin_f64(g.dot(y, Nmy))
    g.pin_f64(lN)

    Pm, lP = Pinv               # current  Phi^-1 (dense), logdet Phi
    Pmr, lPr = Pinv_ref         # reference (constant leaf -> folds)

    # forward covariances for the covariance-space increment dPhi (prior is
    # well-conditioned, so inverting Phi^-1 back is benign).
    Phi, Phir = g.inv(Pm), g.inv(Pmr)
    dPhi = Phi - Phir

    # current solve -- the expensive Cholesky stays in the working (float32) dtype.
    cf, lS = g.cho_factor(Pm + G)
    u = g.cho_solve(cf, v)
    w = Pm @ u                  # Phi^-1 u

    # reference solve + reference logL: all constant -> folds to an f64 constant.
    cfr, lSr = g.cho_factor(Pmr + G)
    ur = g.cho_solve(cfr, v)
    wr = Pmr @ ur
    logL_ref = g.combine_logp_f64(ytNmy, g.dot(v, ur), [lN, lPr, lSr])

    # quadratic increment dQ = w^T dPhi w_ref, accumulated in f64.
    dQ = g.combine_f64(g.dot(w, dPhi @ wr))

    # logdet increment dLdet = slogdet(I + S0^-1 dPhi G),  S0 = I + Phi_ref G.
    S0 = g.node(lambda P, GG: jnp.eye(GG.shape[0], dtype=GG.dtype) + P @ GG,
                [Phir, G], description='I + Phi_ref G')
    dLdet = g.node(
        lambda S, DG: jnp.linalg.slogdet(
            jnp.eye(DG.shape[0], dtype=DG.dtype) + jnp.linalg.solve(S, DG))[1],
        [S0, dPhi @ G], description='slogdet(I + S0^-1 dPhi G)')

    dlnL = g.node(lambda q, d: 0.5 * q - 0.5 * d, [dQ, dLdet],
                  description='0.5 dQ - 0.5 dLdet')
    logp = g.combine_f64(g.node(lambda r, dl: r + dl, [logL_ref, dlnL],
                                description='logL_ref + dlnL'))


@mm.graph
def woodburykernelsolve(g, y, T, Nsolve, F, Pinv):
    """Return (T^T Σ^-1 y, T^T Σ^-1 T) for Σ = N + F P F^T.

    Used by GlobalLikelihood.conditional to project per-pulsar Woodburys
    onto the global GP basis T.
    """
    Nmy, _ = Nsolve(y)
    NmF, _ = Nsolve(F)
    NmT, _ = Nsolve(T)

    FtNmy = g.dot(NmF, y)   # F.T @ N^-1 y
    FtNmF = g.dot(F, NmF)   # F.T @ N^-1 F
    FtNmT = g.dot(NmF, T)   # F.T @ N^-1 T  (N symmetric, so == (N^-1 F)^T T)

    TtNmy = g.dot(NmT, y)
    TtNmF = g.dot(NmT, F)
    TtNmT = g.dot(NmT, T)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)

    TtSy = TtNmy - TtNmF @ g.cho_solve(cf, FtNmy)
    TtST = TtNmT - TtNmF @ g.cho_solve(cf, FtNmT)

    result = g.pair(TtSy, TtST)


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

    # Single-precision (stage 2): pin the array-wide data term and prior logdet to
    # float64, mirroring woodbury(). Each ytNmys element is (y^T N^-1 y + lN), so
    # pinning the sum_all pulls every per-pulsar quadratic + white-noise logdet --
    # and the y-path N -- into float64, where the cross-pulsar ytNmy cancellation
    # actually bites. The separate Nsolve(F) path stays working dtype, so the big
    # FtNmF Cholesky remains float32. lS is left working dtype on purpose (pinning
    # it would force that Cholesky into float64).
    ytNmy = g.pin_f64(g.sum_all(ytNmys))
    FtNmy = g.hstack(FtNmys)
    FtNmF = g.block_diag(FtNmFs)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)

    lP = g.pin_f64(lP.sum()) # g.node(lambda x: jnp.sum(x), inputs=[lP]) # should be an op

    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, g.cho_solve(cf, FtNmy)), [lP, lS])  # Half A: f64 final combine

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
    # Outer (global GP) prior logdet -> float64. The outer Cholesky logdet lS and
    # total_ld stay working dtype (pinning lS would force the global f64 Cholesky;
    # ytNmy is not pinned here because its cone reaches the inner Cholesky via
    # ytNmy_proj -- the raw accumulation is already pinned at source in jointsolve).
    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, g.cho_solve(cf, FtNmy)),
                              [g.pin_f64(lP.sum()), total_ld, lS])  # Half A: f64 final combine


@mm.graph
def globalwoodbury_fused_refdelta(g, refconst, refincr, Pinv, Pinv_ref):
    """Reference+delta twin of ``globalwoodbury_fused`` -- the OUTER half of the
    fused two-level reference+delta (Piece 2 'Half B'; HD / CURN-with-IRN). See
    docs/design/single_precision/ (nested reference+delta).

    Consumes TWO separate graphs from ``vectorwoodburyjointsolve_refdelta`` (the
    inner rung) -- kept separate ON PURPOSE so the constant group folds away (see
    the note at the rung-1 emit site):

        refconst = [aref_sum, ld_in_ref, btil_ref, Gtil_ref, lN_sum]   # all CONSTANT
        refincr  = [dA, d_ld_in, dbtil, dGtil]                         # all LIVE

    where each is, per the note:
      * aref_sum  -- sum_i a~_ref,i  (reference projected leaf-quadratic, f64)
      * ld_in_ref -- sum_i logdet(I + Phi_ref,in_i G_in,i)  (inner ref logdet)
      * btil_ref  -- (npsr, m_out)            reference b~ per pulsar
      * Gtil_ref  -- (npsr, m_out, m_out)     reference G~ per pulsar (block-diag stack)
      * lN_sum    -- sum_i logdet N_i  (white-noise logdet, fixed -> f64)
      * dA        -- Delta A = sum_i Delta a~_i  (projection increment)
      * d_ld_in   -- inner logdet increment (sum_i slogdet(I + S0_in^-1 dPhi_in G_in))
      * dbtil     -- (npsr, m_out)            increment b~ per pulsar
      * dGtil     -- (npsr, m_out, m_out)     increment G~ per pulsar

    Outer level (the cross-pulsar GW prior, dense for HD / block for CURN):
      Pinv     : current  Phi_gw^-1 leaf -> (Phi_gw^-1, logdet Phi_gw)   (live)
      Pinv_ref : reference Phi_ref,gw^-1 leaf (constant -> folds to f64)

    (There is no current-Phi_gw covariance input: route (b)'s outer-logdet
    increment reuses the Cholesky logdets and the quadratic uses dD_gw = Pm - Pmr,
    so the kernel never needs Phi_gw or any inverse of Pm/Pmr in the live cone.)

    Builds logL_ref ONCE in float64 (folds: all reference quantities are
    constants under fixed white noise) and adds the small O(1) increment
    Delta logL (note sec. 5). The quadratic increment uses the sec. 4 two-perturbation
    form (dD_gw = Pm - Pmr); the outer LOGDET increment is formed by reusing the
    current/reference Cholesky logdets directly, (lP + lS_cf) - (lPr + lSr) -- NOT the
    note's slogdet(I + S0_out^-1 mid) (route b; see that block). Never a
    current-minus-reference subtraction of two large logLs. The expensive *current*
    outer Cholesky stays in the working (float32) dtype; the big-minus-big logdet
    subtraction and the final scalar combination are done in float64 (`combine_f64`).
    """
    (aref_sum, ld_in_ref, btil_ref, Gtil_ref, lN_sum) = (
        refconst[0], refconst[1], refconst[2], refconst[3], refconst[4])
    (dA, d_ld_in, dbtil, dGtil) = (
        refincr[0], refincr[1], refincr[2], refincr[3])

    # stack the per-pulsar projected quantities into the outer (global) space:
    # b~ -> (npsr*m_out,), G~ -> block-diag (npsr*m_out, npsr*m_out).
    flat = lambda v: g.node(lambda x: x.reshape(-1), [v])
    bd   = lambda M: g.node(lambda x: jsp.linalg.block_diag(*x), [M])
    btr  = flat(btil_ref)                     # reference b~ (stacked)
    dbt  = flat(dbtil)                        # increment b~ (stacked)
    Gtr  = bd(Gtil_ref)                       # reference G~ (block-diag)
    dGt  = bd(dGtil)                          # increment G~ (block-diag)
    btil = g.node(lambda a, b: a + b, [btr, dbt])   # current b~ = b~_ref + Delta b~
    Gtil = g.node(lambda a, b: a + b, [Gtr, dGt])   # current G~ = G~_ref + Delta G~

    Pm, lP = Pinv                # current  Phi_gw^-1, logdet Phi_gw (lP live)
    Pmr, lPr = Pinv_ref          # reference (constant leaf -> folds); lPr constant

    # NOTE (route b): the outer covariance Phi_gw and the explicit covariance-space
    # perturbation dPhi_gw are no longer built. They existed only to feed the
    # two-perturbation outer-logdet form (mid/S0_out/slogdet); that whole block is
    # replaced below by reusing the current/reference Cholesky logdets directly. The
    # quadratic increment routes through dD_gw = Pm - Pmr (see below), which needs
    # neither Phi_gw nor inv(Pmr) -- so inv(Pm)/inv(Pmr) never appear in the live
    # cone, and the kernel no longer takes a current-Phi_gw covariance input.

    # --- current outer solve -- the expensive Cholesky stays float32 ---
    # KEEP its logdet lS_cf (current outer capacitance logdet, was discarded):
    # route (b) reuses it for the outer-logdet increment instead of a slogdet LU.
    cf, lS_cf = g.cho_factor(Pm + Gtil)
    nu = g.cho_solve(cf, btil)                # nu = C_out^-1 b~

    # --- reference outer solve + reference logL: all constant -> folds to f64 ---
    cfr, lSr = g.cho_factor(Pmr + Gtr)
    nu_ref = g.cho_solve(cfr, btr)            # nu_ref = C_ref,out^-1 b~_ref
    quad_ref = g.dot(btr, nu_ref)             # b~_ref . nu_ref
    # logL_ref = -0.5(a~_ref - b~_ref.nu_ref) - 0.5(lN + ld_in_ref + lP_ref,gw + lS_ref,gw)
    logL_ref = g.combine_logp_f64(aref_sum, quad_ref,
                                  [lN_sum, ld_in_ref, lPr, lSr])

    # --- outer quadratic increment (note sec. 4, two-perturbation) ---
    # dK_out = dD_gw + Delta G~. The note routes dD_gw = -Pm dPhi_gw Pmr, but with
    # dPhi_gw = Phi_gw - Phi_ref,gw = Pm^-1 - Pmr^-1 that collapses ALGEBRAICALLY to
    #   -Pm (Pm^-1 - Pmr^-1) Pmr = -(Pm Pm^-1) Pmr + Pm (Pmr^-1 Pmr) = Pm - Pmr,
    # turning two dense (npsr*m_gw)^3 matmuls into one subtraction. The product form
    # existed only to dodge the Pm-Pmr cancellation (both ~1e10); we recover that
    # safety by doing the subtraction in f64 (combine_f64, operands cast up on read).
    dD_gw = g.combine_f64(g.node(lambda P, Pr: P - Pr, [Pm, Pmr],
                                 description='dD_gw = Pm - Pmr (= -Pm dPhi_gw Pmr)'))
    dK = g.node(lambda a, b: a + b, [dD_gw, dGt], description='dK_out = dD_gw + Delta G~')
    # dQ_out = Delta b~ . nu + b~_ref . C_out^-1 (Delta b~ - dK nu_ref)
    cross = g.cho_solve(cf, g.node(lambda db, K, nr: db - K @ nr, [dbt, dK, nu_ref],
                                   description='Delta b~ - dK nu_ref'))
    dQ = g.combine_f64(g.node(lambda db, nu_, br, cr: db @ nu_ + br @ cr,
                              [dbt, nu, btr, cross],
                              description='dQ_out = Delta b~.nu + b~_ref.C^-1(Delta b~ - dK nu_ref)'))

    # --- outer logdet increment (route b): reuse the Cholesky logdets ---
    # The full outer logdet (cf. globalwoodbury_fused) is lP + lS, i.e.
    #   logdet Phi_gw + logdet(Phi_gw^-1 + G~) = logdet(I + Phi_gw G~),
    # so the increment is (lP + lS_cf) - (lPr + lSr): the live current pieces minus
    # the (folded, constant) reference pieces. This DELETES the dense mid/S0_out
    # matmuls and the dense slogdet LU -- the current outer factorization `cf` (which
    # we already pay for `nu`) now supplies the logdet too, so refdelta's dense work
    # collapses to ~the normal path's single Cholesky. Cost: this is a big-minus-big
    # (each side ~1e4, increment O(1)) and lS_cf is only f32-accurate (born from the
    # f32 Cholesky), so do the subtraction in f64 (combine_f64) -- lP/lPr/lSr are
    # already f64. Accuracy floor moves ~5e-4 -> ~1e-3 (CPU), still well under the
    # GPU TF32 floor (~1.2e-2) where this f32 path actually runs.
    d_ld_out = g.combine_f64(
        g.node(lambda lp, ls, lpr, lsr: (lp + ls) - (lpr + lsr),
               [lP, lS_cf, lPr, lSr],
               description='d_ld_out = (lP + lS_cf) - (lPr + lSr)'))

    # --- assembly (note sec. 5): Delta logL = -0.5[(dA - dQ) + (d_ld_in + d_ld_out)] ---
    dlnL = g.node(lambda da, q, ldi, ldo: -0.5 * ((da - q) + (ldi + ldo)),
                  [dA, dQ, d_ld_in, d_ld_out], description='Delta logL')
    logp = g.combine_f64(g.node(lambda r, dl: r + dl, [logL_ref, dlnL],
                                description='logL_ref + Delta logL'))


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
        # Single-precision (stage 2): pin the per-pulsar white-noise logdet to
        # float64 at source (mirrors woodbury). The downstream combine into `ld`
        # stays working dtype -- that is Piece 2's job, not the pins'.
        lNs.append(g.pin_f64(lN))
        Nmys.append(Nmy)
        NmFs_out.append(NmF_out)
        NmFs_in.append(NmF_in)
        FtNmys_in.append(g.dot(F_in, Nmy))
        FtNmFs_in.append(g.dot(F_in, NmF_in))

    FtNmy_in = g.array(FtNmys_in)
    FtNmF_in = g.array(FtNmFs_in)

    Pm, lP = Pinv
    lP = g.pin_f64(lP)   # inner GP prior logdet -> float64 (pinned at source)
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
    # Pin each per-pulsar raw y^T N^-1 y to float64 at source (mirrors woodbury's
    # ytNmy pin). Cone = Nsolve(y) -> N only; the inner Cholesky cf is the *other*
    # operand of the ytNmy_proj subtraction, not an ancestor, so it stays float32.
    ytNmy_consts = g.array([g.pin_f64(g.dot(ys[i], Nmys[i])) for i in range(len(ys))])  # (67,)
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
def vectorwoodburyjointsolve_refdelta(g, ys, Fs_outer, Nsolves, Fs_inner, Pinv, Pinv_ref):
    """Reference+delta twin of ``vectorwoodburyjointsolve`` -- the INNER half of the
    fused two-level reference+delta (Piece 2 'Half B'; HD / CURN-with-IRN). See
    docs/design/single_precision/ (nested reference+delta).

    Same per-pulsar inner intrinsic-red-noise (IRN) solve as
    ``vectorwoodburyjointsolve``, but off a frozen inner reference covariance
    Phi_ref,in (``Pinv_ref``) it emits, for the outer level (``globalwoodbury_fused_refdelta``):

      * the REFERENCE projected quantities (a~_ref summed, b~_ref stacked,
        G~_ref) -- constant under fixed white noise, so they fold to float64;
      * the per-call INCREMENTS (Delta a~, Delta b~, Delta G~) formed DIRECTLY via
        the resolvent identity Delta mu = -C^-1 Delta D mu_ref (NEVER as a
        current-minus-reference subtraction of two large numbers -- that is the
        catastrophic float32 cancellation reference+delta exists to avoid);
      * the inner reference logdet (sum_i logdet(I + Phi_ref,in_i G_in_i)) and its
        per-call increment (sum_i slogdet(I + S0_in^-1 dPhi_in G_in)).

    Inner Phi is diagonal (power-law IRN), so Delta D is formed per mode as
    -dphi/(phi phi_ref) -- no inverse-difference cancellation (note sec. 2, sec. 5).

    Inputs mirror ``vectorwoodburyjointsolve`` plus ``Pinv_ref`` (frozen inner
    reference Phi_ref,in^-1 leaf -> folds to float64 constants).
    """
    Nmys, NmFs_out, NmFs_in = [], [], []
    FtNmys_in, FtNmFs_in = [], []
    lNs = []

    for y, F_out, F_in, Nsolve in zip(ys, Fs_outer, Fs_inner, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF_out, _ = Nsolve(F_out)
        NmF_in, _ = Nsolve(F_in)
        lNs.append(g.pin_f64(lN))               # white-noise logdet (fixed -> f64)
        Nmys.append(Nmy)
        NmFs_out.append(NmF_out)
        NmFs_in.append(NmF_in)
        FtNmys_in.append(g.dot(F_in, Nmy))      # b_in,i  (fixed)
        FtNmFs_in.append(g.dot(F_in, NmF_in))   # G_in,i  (fixed)

    FtNmy_in = g.array(FtNmys_in)               # (npsr, m_in)        = b_in
    G_in = g.array(FtNmFs_in)                   # (npsr, m_in, m_in)  = G_in
    H = g.array([g.dot(F_in, NmF_out) for F_in, NmF_out in zip(Fs_inner, NmFs_out)])  # (npsr, m_in, m_out)

    # fixed per-pulsar leaf data terms / outer-basis consts (no Phi dependence)
    a_leaf = g.pin_f64(g.sum_all([g.pin_f64(g.dot(ys[i], Nmys[i])) for i in range(len(ys))]))  # sum_i a_i
    FtNmy_out = g.array([g.dot(Fs_outer[i], Nmys[i]) for i in range(len(ys))])        # (npsr, m_out)        = b_out
    Ht = g.array([g.dot(Fs_outer[i], NmFs_in[i]) for i in range(len(ys))])            # (npsr, m_out, m_in)  = H^T
    FtNmF_out = g.array([g.dot(Fs_outer[i], NmFs_out[i]) for i in range(len(ys))])    # (npsr, m_out, m_out) = G_out

    # --- current inner solve (the expensive batched Cholesky stays float32) ---
    Pm, lP = Pinv
    cf, _ = g.cho_factor(Pm + G_in)

    # --- reference inner solve (all constant -> folds to a float64 constant) ---
    Pmr, _ = Pinv_ref
    cfr, _ = g.cho_factor(Pmr + G_in)
    mu_y_ref = g.cho_solve(cfr, FtNmy_in)       # (npsr, m_in)        = C_ref^-1 b_in
    mu_F_ref = g.cho_solve(cfr, H)              # (npsr, m_in, m_out) = C_ref^-1 H

    # --- safe inner Delta D (per mode; diagonal inner Phi) and dphi ---
    # Pm, Pmr are batched diagonal matrices (npsr, m, m); work on their diagonals.
    diag = lambda M: jnp.diagonal(M, axis1=-2, axis2=-1)
    phi = g.node(lambda P: 1.0 / diag(P), [Pm], description='inner phi = 1/diag(Pm)')        # (npsr, m_in)
    phir = g.node(lambda P: 1.0 / diag(P), [Pmr], description='inner phi_ref = 1/diag(Pmr)')
    dphi = g.node(lambda a, b: a - b, [phi, phir], description='inner dphi = phi - phi_ref')  # safe (covariance space)
    dD = g.node(lambda dp, a, b: -dp / (a * b), [dphi, phi, phir],
                description='inner Delta D = 1/phi - 1/phi_ref (safe, per mode)')             # (npsr, m_in)

    # inner increments via the resolvent: Delta mu = -C^-1 (Delta D mu_ref)  (Delta D diagonal)
    dDmu_y = g.node(lambda d, m: d * m, [dD, mu_y_ref], description='dD mu_y_ref')             # (npsr, m_in)
    dDmu_F = g.node(lambda d, m: d[:, :, None] * m, [dD, mu_F_ref], description='dD mu_F_ref')  # (npsr, m_in, m_out)
    dmu_y = g.node(lambda x: -x, [g.cho_solve(cf, dDmu_y)],
                   description='inner Delta mu_y = -C^-1 dD mu_y_ref')                         # (npsr, m_in)
    dmu_F = g.node(lambda x: -x, [g.cho_solve(cf, dDmu_F)],
                   description='inner Delta mu_F = -C^-1 dD mu_F_ref')                         # (npsr, m_in, m_out)

    # --- reference projected quantities (fold to f64) ---  (note sec. 0 projection)
    aref = g.node(lambda b, m: jnp.einsum('pi,pi->p', b, m), [FtNmy_in, mu_y_ref])            # b_in . mu_y_ref
    aref_sum = g.combine_f64(g.node(lambda al, x: al - jnp.sum(x), [a_leaf, aref],
                                    description='sum_i a~_ref,i = sum_i (a_i - b_in.mu_y_ref)'))
    btil_ref = g.node(lambda bo, A, m: bo - jnp.einsum('pij,pj->pi', A, m), [FtNmy_out, Ht, mu_y_ref])   # (npsr, m_out)
    Gtil_ref = g.node(lambda Go, A, m: Go - jnp.einsum('pij,pjk->pik', A, m), [FtNmF_out, Ht, mu_F_ref])  # (npsr, m_out, m_out)

    # --- projected increments (formed from the inner Delta mu; no big-minus-big) --- (note sec. 3)
    dA = g.combine_f64(g.node(lambda b, dm: -jnp.sum(jnp.einsum('pi,pi->p', b, dm)), [FtNmy_in, dmu_y],
                              description='Delta A = sum_i Delta a~_i = -sum_i b_in . Delta mu_y'))
    dbtil = g.node(lambda A, dm: -jnp.einsum('pij,pj->pi', A, dm), [Ht, dmu_y],
                   description='Delta b~ = -H^T Delta mu_y')                                   # (npsr, m_out)
    dGtil = g.node(lambda A, dm: -jnp.einsum('pij,pjk->pik', A, dm), [Ht, dmu_F],
                   description='Delta G~ = -H^T Delta mu_F')                                   # (npsr, m_out, m_out)

    # --- inner logdet: reference (folds f64) + increment --- (note sec. 4, batched per pulsar)
    # S0_in,i = I + Phi_ref,in_i G_in,i (Phi_ref diagonal -> phir[:, :, None] * G_in)
    S0_in = g.node(lambda pr, G: jnp.eye(G.shape[-1], dtype=G.dtype) + pr[:, :, None] * G,
                   [phir, G_in], description='inner S0 = I + Phi_ref,in G_in')                 # (npsr, m_in, m_in)
    ld_in_ref = g.pin_f64(g.node(lambda S: jnp.sum(jnp.linalg.slogdet(S)[1]), [S0_in],
                                 description='sum_i logdet(I + Phi_ref,in G_in)'))
    d_ld_in = g.combine_f64(g.node(
        lambda S, dp, G: jnp.sum(jnp.linalg.slogdet(
            jnp.eye(G.shape[-1], dtype=G.dtype)
            + jnp.linalg.solve(S, dp[:, :, None] * G))[1]),
        [S0_in, dphi, G_in], description='sum_i slogdet(I + S0_in^-1 dPhi_in G_in)'))

    # white-noise logdet (fixed; cancels in Delta logL but needed once so the
    # outer rung can build the absolute logL_ref that matches globalwoodbury_fused).
    lN_sum = g.pin_f64(g.sum_all(lNs))

    # Emit the reference (constant) and increment (live) quantities as TWO
    # SEPARATE named outputs, NOT one bundled tuple. This is load-bearing for
    # performance: the outer rung consumes each as its own graph. If they share a
    # single output tuple (hence a single GraphLeaf), then because the increments
    # are live the *whole* leaf is live, and the outer fold_constants can no longer
    # see that the reference quantities (Gtil_ref, btil_ref, ...) are constant ->
    # the outer reference Cholesky / inv never fold and get recomputed every call,
    # scaling with n_psr. Split, the `refconst` graph has only constant outputs, so
    # it folds wholesale to frozen leaves the outer graph reads as constants.
    g.named(g.ntuple([aref_sum, ld_in_ref, btil_ref, Gtil_ref, lN_sum]), 'refconst')
    g.named(g.ntuple([dA, d_ld_in, dbtil, dGtil]), 'refincr')


@mm.graph
def vectorwoodbury(g, ys, Nsolves, Fs, Pinv):
    ytNmys, FtNmys, FtNmFs = [], [], []

    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)

        ytNmys.append(y.T @ Nmy + lN)
        FtNmys.append(F.T @ Nmy)
        FtNmFs.append(F.T @ NmF)

    # Single-precision (stage 2): same pins as globalwoodbury(). Each ytNmys
    # element is (y^T N^-1 y + lN); pin the array sum_all to float64 (the
    # cross-pulsar cancellation term) and the prior logdet sum. lS stays working
    # dtype so the batched FtNmF Cholesky remains float32.
    ytNmy = g.pin_f64(g.sum_all(ytNmys))
    FtNmy = g.array(FtNmys)
    FtNmF = g.array(FtNmFs)

    Pm, lP = Pinv
    cf, lS = g.cho_factor(Pm + FtNmF)
    mu = g.cho_solve(cf, FtNmy)

    cond = g.pair(mu, cf, name='cond')
    logp = g.combine_logp_f64(ytNmy, g.sum(FtNmy * mu), [g.pin_f64(lP.sum()), lS.sum()])  # Half A: f64 final combine


@mm.graph
def vectorwoodbury_refdelta(g, ys, Nsolves, Fs, Pinv, Pinv_ref):
    """Reference+delta twin of ``vectorwoodbury`` -- single GP level, batched over
    pulsars (Piece 2 'Half B'). The production form of the validated scalar
    ``woodbury_refdelta`` for **no-Hellings-Downs array analyses** (per-pulsar IRN,
    CURN, IRN+CURN via ``make_combined_crn``): each pulsar carries one red-noise GP
    with a diagonal Phi and there is no cross-pulsar coupling, so the array
    likelihood factorises, logL = sum_i logL_i. We therefore expand each pulsar's
    logL about a frozen reference covariance Phi_ref,i and sum:

        logL(theta) = sum_i logL_ref,i  +  sum_i Delta logL,i,

    with sum_i logL_ref,i computed ONCE in float64 (folds, fixed white noise) and
    only the O(1) increment per call. The increment is formed directly (note sec.
    2-3) -- never as the float32 cancellation of two ~1e6 numbers -- and the
    expensive batched GP-block Cholesky stays float32 (the speed win).

    Inputs mirror ``vectorwoodbury`` plus ``Pinv_ref`` (frozen reference Phi_ref^-1
    leaf -> folds to float64 constants). See research_note_on_split_with_reference.md
    (sec. 2-3), docs/adr/0001,0003, and the HD/coupled case in
    ``vectorwoodburyjointsolve_refdelta`` + ``globalwoodbury_fused_refdelta``.
    """
    ytNmys, FtNmys, FtNmFs = [], [], []
    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)
        ytNmys.append(y.T @ Nmy + lN)        # folds lN in, mirroring vectorwoodbury
        FtNmys.append(g.dot(F, Nmy))         # b_i = F^T N^-1 y
        FtNmFs.append(g.dot(F, NmF))         # G_i = F^T N^-1 F

    # ytNmy = sum_i (y^T N^-1 y + lN) -- the cross-pulsar cancellation term, pin f64.
    ytNmy = g.pin_f64(g.sum_all(ytNmys))
    v = g.array(FtNmys)                       # (npsr, m)
    G = g.array(FtNmFs)                       # (npsr, m, m)

    Pm, _ = Pinv                              # current  Phi^-1 (batched diagonal), live
    Pmr, lPr = Pinv_ref                       # reference Phi_ref^-1 (constant -> folds f64)

    # covariance-space increment dPhi = Phi - Phi_ref (prior well-conditioned -> benign)
    Phi, Phir = g.inv(Pm), g.inv(Pmr)
    dPhi = g.node(lambda a, b: a - b, [Phi, Phir], description='dPhi = Phi - Phi_ref')

    # current solve -- the batched GP-block Cholesky stays float32.
    cf, _ = g.cho_factor(Pm + G)
    u = g.cho_solve(cf, v)
    w = g.node(lambda P, x: jnp.einsum('pij,pj->pi', P, x), [Pm, u], description='w = Phi^-1 u')

    # reference solve + reference logL: all constant -> folds to an f64 constant.
    cfr, lSr = g.cho_factor(Pmr + G)
    ur = g.cho_solve(cfr, v)
    wr = g.node(lambda P, x: jnp.einsum('pij,pj->pi', P, x), [Pmr, ur], description='w_ref = Phi_ref^-1 u_ref')
    logL_ref = g.combine_logp_f64(ytNmy, g.sum(v * ur), [g.pin_f64(lPr.sum()), lSr.sum()])

    # quadratic increment dQ = sum_i w_i^T dPhi_i w_ref_i, accumulated in f64.
    dQ = g.combine_f64(g.node(lambda W, D, Wr: jnp.sum(jnp.einsum('pi,pij,pj->p', W, D, Wr)),
                              [w, dPhi, wr], description='dQ = sum_i w dPhi w_ref'))

    # logdet increment sum_i slogdet(I + S0_i^-1 dPhi_i G_i),  S0 = I + Phi_ref G.
    S0 = g.node(lambda P, GG: jnp.eye(GG.shape[-1], dtype=GG.dtype) + P @ GG, [Phir, G],
                description='S0 = I + Phi_ref G')
    dPhiG = g.node(lambda D, GG: D @ GG, [dPhi, G], description='dPhi G')
    dLdet = g.combine_f64(g.node(
        lambda S, DG: jnp.sum(jnp.linalg.slogdet(
            jnp.eye(DG.shape[-1], dtype=DG.dtype) + jnp.linalg.solve(S, DG))[1]),
        [S0, dPhiG], description='sum_i slogdet(I + S0^-1 dPhi G)'))

    dlnL = g.node(lambda q, d: 0.5 * q - 0.5 * d, [dQ, dLdet], description='0.5 dQ - 0.5 dLdet')
    logp = g.combine_f64(g.node(lambda r, dl: r + dl, [logL_ref, dlnL],
                                description='logL_ref + dlnL'))


@mm.graph
def gaussian_coefficient_logprior(g, c_for_prior, Pinv):
    """Gaussian log-prior on per-pulsar GP coefficients: `-0.5 c.T Pm c - 0.5 sum(ldP)`.

    General-purpose; not tied to any particular compound topology. Used as the
    `prior` leaf of `vectorgpcomponent` whenever the prior precision collapses
    into a single `Pinv` leaf.

    Inputs:
      c_for_prior : (npsr, k)        — per-pulsar coefficient matrix.
      Pinv        : GraphLeaf returning (Pm, ldP) with
                      Pm  shape (npsr, k, k)  — batched precision per pulsar
                                                (batched-diagonal under
                                                 metamatrix.matrix_inv);
                      ldP shape (npsr,)       — per-pulsar log|Phi|.

    Output:
      scalar `logpr = -0.5 * einsum('ij,ijk,ik->', c, Pm, c) - 0.5 * sum(ldP)`.
    """
    Pm, ldP = Pinv
    quad = g.node(
        lambda c_, P_: jnp.einsum('ij,ijk,ik->', c_, P_, c_),
        [c_for_prior, Pm], description='c_for_prior.T @ Pm @ c_for_prior')
    logpr = -0.5 * quad - 0.5 * g.sum(ldP)


@mm.graph
def vectorgpcomponent(g, ys, Nsolves, Fs, prior, coeffs, means, ext_coeffs, ext_Fs):
    """ArrayLikelihood.clogL: log p(y, c) at fixed GP coefficients c.

    Mirrors `vectorwoodbury`'s per-pulsar precompute, but the GP coefficients
    come from outside (a FuncLeaf folding params -> (c, ldL)) instead of being
    solved for as the MAP. The prior on c is centered on `means(params)` when
    provided. ExtSignals contribute their own basis cross-terms.

    Leaves:
      ys         : per-pulsar residual ArgLeafs.
      Nsolves    : per-pulsar `N.make_solve` GraphLeafs (each (y) -> (Nmy, lN)).
      Fs         : per-pulsar GP basis (Const/Func/GraphLeafs).
      prior      : GraphLeaf `(c_for_prior) -> scalar logpr`. The single-Pinv
                   case wraps `P.make_inv` via `gaussian_coefficient_logprior`;
                   the mixed-Phi case (commongp + globalgp(HD)) supplies a
                   per-GP-sum graph built by `CompoundGP._build_mixed_logprior`.
                   Same interface.
      coeffs     : FuncLeaf params -> (c, ldL); c shape (npsr, k); ldL is the
                   summed log-|J| from reparams.
      means      : FuncLeaf params -> a0 of shape (npsr, k), or ConstLeaf(0.0)
                   if no means are configured.
      ext_coeffs : list of FuncLeafs (one per ExtSignal): params -> ccw.
      ext_Fs     : list of lists of per-pulsar Fcw leaves (one list per ExtSignal).

    Named outputs:
      'logp'   — scalar log-likelihood.
      'staged' — pair (logp, c); used when reparams were applied so the caller
                 can see the post-reparam coefficients.
    """
    NmFs, ldNs, ytNmys, NmFtys, FtNmFs = [], [], [], [], []
    for y, F, Nsolve in zip(ys, Fs, Nsolves):
        Nmy, lN = Nsolve(y)
        NmF, _ = Nsolve(F)
        NmFs.append(NmF)
        ldNs.append(lN)
        ytNmys.append(g.dot(y, Nmy))
        NmFtys.append(g.dot(NmF, y))   # F.T @ N^-1 y
        FtNmFs.append(g.dot(F, NmF))   # F.T @ N^-1 F

    ytNmy = g.sum_all(ytNmys)
    ldN   = g.sum_all(ldNs)
    NmFty = g.array(NmFtys)            # (npsr, k)
    FtNmF = g.array(FtNmFs)            # (npsr, k, k)

    c, ldL = coeffs.split()
    c_for_prior = c - means            # means is ConstLeaf(0.0) when absent

    logpr = prior(c_for_prior)         # Apply on GraphLeaf -> scalar

    quad_c = g.node(
        lambda c_, F_: jnp.einsum('ij,ijk,ik->', c_, F_, c_),
        [c, FtNmF], description='c.T @ FtNmF @ c')
    cNmFty = g.node(lambda c_, n_: jnp.sum(c_ * n_), [c, NmFty],
                    description='sum(c * NmFty)')
    data_term = -0.5 * ytNmy + cNmFty - 0.5 * quad_c - 0.5 * ldN

    logp = data_term + logpr + ldL
    for ccw, Fcw_list in zip(ext_coeffs, ext_Fs):
        NmFcws = [Nsolve(Fcw_i) for Nsolve, Fcw_i in zip(Nsolves, Fcw_list)]
        FcwNmy    = g.array([g.dot(NmFcw[0], y) for NmFcw, y in zip(NmFcws, ys)])
        FtNmFcw   = g.array([g.dot(F, NmFcw[0]) for F, NmFcw in zip(Fs, NmFcws)])
        FcwtNmFcw = g.array([g.dot(Fcw_i, NmFcw[0])
                             for Fcw_i, NmFcw in zip(Fcw_list, NmFcws)])
        cwdata = g.node(lambda x_, B_: jnp.sum(x_ * B_), [ccw, FcwNmy],
                        description='sum(ccw * FcwNmy)')
        cross = g.node(lambda c_, A_, x_: jnp.einsum('ij,ijk,ik->', c_, A_, x_),
                       [c, FtNmFcw, ccw], description='c.T @ FtNmFcw @ ccw')
        cwself = g.node(lambda x_, A_: jnp.einsum('ij,ijk,ik->', x_, A_, x_),
                        [ccw, FcwtNmFcw], description='ccw.T @ FcwtNmFcw @ ccw')
        logp = logp + cwdata - cross - 0.5 * cwself

    g.named(logp, 'logp')
    g.named(g.pair(logp, c), 'staged')


@mm.graph
def vectorresidualcomponent(g, ys, Nsolves, Fs, prior, coeffs, means,
                            ext_coeffs, ext_Fs):
    """ArrayLikelihood.clogL, residual form: log p(y, c) at fixed GP
    coefficients with NO F^T N^-1 F products.

    Identical leaf contract to `vectorgpcomponent` (same inputs, same named
    outputs 'logp'/'staged', same prior/means/reparam semantics); different
    algebra:

        r_i  = y_i - F_i @ c[i] - sum_e Fext_e,i @ ccw_e[i]
        logp = sum_i [ -0.5 r_i^T N_i^-1 r_i - 0.5 logdet N_i ]
               + prior(c - means) + ldL

    With constant N the per-pulsar solve factorization folds at trace time and
    each evaluation costs one O(n_toa * k) matvec plus a vector solve. With
    parameter-dependent N nothing of shape (n_toa, k) is ever pushed through
    the solve -- this is the varying-white-noise fast path.

    ExtSignal contributions are plain subtractions from `r`; the cross-term
    algebra of `vectorgpcomponent` is NOT replicated -- the two forms are
    algebraically identical.

    No f64 pins in this graph: it is float64-default; single-precision
    treatment is out of scope here.
    """
    c, ldL = coeffs.split()
    logpr = prior(c - means)           # means is ConstLeaf(0.0) when absent

    data_terms, ldNs = [], []
    for i, (y, F, Nsolve) in enumerate(zip(ys, Fs, Nsolves)):
        r = y - F @ c[i]
        for ccw, Fcw_list in zip(ext_coeffs, ext_Fs):
            r = r - Fcw_list[i] @ ccw[i]
        Nmr, lN = Nsolve(r)
        data_terms.append(g.dot(r, Nmr))
        ldNs.append(lN)

    logp = (-0.5 * g.sum_all(data_terms) - 0.5 * g.sum_all(ldNs)
            + logpr + ldL)

    g.named(logp, 'logp')
    g.named(g.pair(logp, c), 'staged')


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



def _materialize(x):
    """Resolve x — array, callable, or metamath graph — to (const_array, fn).

    Exactly one of the returned values is non-None. `fn` (when set) has a
    `.params` attribute and is called as `fn(params=params)`.
    """
    if isinstance(x, dict):           # metamath graph
        fn = mm.func(x)
        if not fn.args and not fn.params:
            return jnp.asarray(fn(params={})), None
        return None, fn
    if callable(x):
        return None, x
    return jnp.asarray(x), None


def _sm_apply(y, Np, Uind, P):
    """Apply K^-1 to y for K = diag(N) + F P F^T (exposure-indexed Sherman-Morrison).

    Dispatches on y.ndim so the same node serves both `Nsolve(y)` (1D) and
    `Nsolve(F)` (2D) call-sites in the Woodbury graph. Takes the already-
    padded Np (so `pad(N)` can fold when N is constant).
    """
    if y.ndim == 1:
        yp = jnp.pad(y, ((1, 0),), constant_values=0.0)
        return smup_ind_correct(yp, Np, Uind, P)
    else:
        # y has shape (d, k); vsmup_ind_correct expects (k, d+1).
        Tp = jnp.pad(y.T, ((0, 0), (1, 0)), constant_values=0.0)
        return vsmup_ind_correct(Tp, Np, Uind, P).T


@mm.graph
def smsolve(g, y, N, Uind, P):
    """Sherman-Morrison-Woodbury solve `K^-1 y` for `K = diag(N) + F P F^T`.

    F is the ecorr exposure matrix (0/1 epoch indicators), encoded compactly
    as ``Uind`` (per-epoch list of TOA indices, padded with 0 to a common
    length; index 0 of Np is a sentinel +inf so padded entries contribute 0).

    The math is decomposed across graph nodes so the constant pieces fold:

      ``logN``   = ``sum(log N)``           — folds when N is const
      ``Np``     = ``pad(N)`` with +inf     — folds when N is const
      ``log1pt`` = ``sum log1p(P · F^T diag(1/N) F)`` (the F^T·F is
                   diagonal by epoch, gathered via ``Uind``)
                                            — folds when both N and P are const
      ``Kmy``    = ``K^-1 y``               — runtime (depends on the
                                              residual / basis matrix)

    Returns (Kmy, logdet). Equivalent to ``matrix.SM_1d_indexed`` (1D y) /
    ``matrix.SM_2d_indexed`` (2D y) but exposed as a graph so folding can
    bake the N-only and (N,P)-only precompute when noisedict / ecorr-prior
    are pinned.
    """
    logN = g.node(lambda N_: jnp.sum(jnp.log(N_)),
                  [N], description='sum(log N)')
    Np = g.node(lambda N_: jnp.pad(N_, ((1, 0),), constant_values=jnp.inf),
                [N], description='pad(N) with +inf at idx 0')
    log1pt = g.node(lambda Np_, P_, U_: jnp.sum(vsmdp_ind(Np_, P_, U_)),
                    [Np, P, Uind],
                    description='sum log1p(P · F^T diag(1/N) F)')
    logdet = logN + log1pt

    Kmy = g.node(_sm_apply, [y, Np, Uind, P], description='K^-1 y')
    result = g.pair(Kmy, logdet)


def _sm_whiten_apply(y, Np, Uind, P):
    """Apply the whitening W = K^{-1/2} to y for K = diag(N) + F P F^T.

    Same exposure-indexed Sherman-Morrison structure and ndim dispatch as
    ``_sm_apply``, but returns W y (not K^-1 y). Used by the timing-model
    projection; see adr/0004-timing-model-projection.md.
    """
    if y.ndim == 1:
        yp = jnp.pad(y, ((1, 0),), constant_values=0.0)
        return smwhiten_ind_correct(yp, Np, Uind, P)
    else:
        Tp = jnp.pad(y.T, ((0, 0), (1, 0)), constant_values=0.0)
        return vsmwhiten_ind_correct(Tp, Np, Uind, P).T


@mm.graph
def smwhiten(g, y, N, Uind, P):
    """Whitening `W y` with `W = K^{-1/2}`, `K = diag(N) + F P F^T` (ECORR).

    Parallel to ``smsolve`` but applies the inverse *square root* instead of the
    inverse. Same fold structure (``logN``, ``Np``, ``log1pt`` bake when N / P are
    pinned). Returns (Wy, logdet K). For diagonal-only noise this reduces to
    ``y / sqrt(N)`` with ``logdet = sum(log N)`` (no epochs).
    """
    logN = g.node(lambda N_: jnp.sum(jnp.log(N_)),
                  [N], description='sum(log N)')
    Np = g.node(lambda N_: jnp.pad(N_, ((1, 0),), constant_values=jnp.inf),
                [N], description='pad(N) with +inf at idx 0')
    log1pt = g.node(lambda Np_, P_, U_: jnp.sum(vsmdp_ind(Np_, P_, U_)),
                    [Np, P, Uind],
                    description='sum log1p(P · F^T diag(1/N) F)')
    logdet = logN + log1pt

    Wy = g.node(_sm_whiten_apply, [y, Np, Uind, P], description='K^{-1/2} y')
    result = g.pair(Wy, logdet)


def _diag_whiten_apply(y, N):
    """Apply W = diag(N)^{-1/2} to y (1d residual or 2d basis), ndim-dispatched
    like ``_sm_whiten_apply`` so one node serves both call sites."""
    s = jnp.sqrt(N)
    if y.ndim == 1:
        return y / s
    else:
        return y / s[:, None]


@mm.graph
def dwhiten(g, y, N):
    """Whitening `W y` with `W = diag(N)^{-1/2}` for purely diagonal noise.

    The diagonal counterpart of ``smwhiten`` (no ECORR epochs): returns
    (y / sqrt(N), sum(log N)). Used by the timing-model projection for PTAs
    without ECORR; see adr/0004-timing-model-projection.md.
    """
    logN = g.node(lambda N_: jnp.sum(jnp.log(N_)),
                  [N], description='sum(log N)')
    Wy = g.node(_diag_whiten_apply, [y, N], description='N^{-1/2} y')
    result = g.pair(Wy, logN)


class NoiseMatrixSM(Kernel):
    """metamath analog of matrix.NoiseMatrixSM_var.

    K = diag(N) + F P F^T where F is the ecorr exposure matrix (0/1 indicators
    per epoch). Uses the indexed Sherman-Morrison solve via a graph node.
    """
    def __init__(self, N, F, P):
        # signature matches matrix.NoiseMatrixSM_var(getnoise, F, getP)
        self.N = N            # callable: params -> diag noise (n_toa,)
        self.F = F            # exposure matrix (n_toa, n_epoch), constant
        self.P = P            # callable: params -> per-epoch variance (n_epoch,)
        self.Uind = jnp.array(make_uind(F))

    @property
    def make_solve(self):
        return smsolve(None, self.N, self.Uind, self.P)

    @property
    def make_whiten(self):
        # K^{-1/2} applicator for the timing-model projection.
        return smwhiten(None, self.N, self.Uind, self.P)

    @property
    def make_inv(self):
        # not needed for Woodbury, but provide for completeness via dense fallback
        raise NotImplementedError("NoiseMatrixSM.make_inv not implemented; "
                                  "use make_solve inside a Woodbury.")

    def make_kernelproduct(self, y):
        return normal(y, self.make_solve)

    def make_sample(self):
        """Sample noise ~ N(0, K) where K = diag(N) + F P F^T.

        Equivalent to drawing N(0, diag(N)) + F @ N(0, diag(P)).
        """
        N_const, N_fn = _materialize(self.N)
        P_const, P_fn = _materialize(self.P)
        F = jnp.asarray(self.F)

        def sample(key, params={}):
            N_arr = N_const if N_fn is None else N_fn(params=params)
            P_arr = P_const if P_fn is None else P_fn(params=params)

            key, k1, k2 = jax.random.split(key, 3)
            noise = jax.random.normal(k1, N_arr.shape) * jnp.sqrt(N_arr)
            ecorr = jax.random.normal(k2, P_arr.shape) * jnp.sqrt(P_arr)
            return key, noise + F @ ecorr

        sample.params = sorted(set(
            (list(N_fn.params) if N_fn is not None else []) +
            (list(P_fn.params) if P_fn is not None else [])
        ))
        return sample


class NoiseMatrix(Kernel):
    def __init__(self, N):
        self.N = N
        # matrix.NoiseMatrix*_var classes expose the callable as `getN`; alias
        # for compat with signals.py construction (e.g. NoiseMatrixSM_var(..., egp.Phi.getN))
        self.getN = N

    def make_sample(self):
        """Sample x ~ N(0, N). Dispatches on N.ndim at runtime: 1D → diag, 2D → cholesky."""
        N_const, N_fn = _materialize(self.N)

        def sample(key, params={}):
            N_arr = N_const if N_fn is None else N_fn(params=params)
            key, subkey = jax.random.split(key)
            if N_arr.ndim == 1:
                return key, jax.random.normal(subkey, N_arr.shape) * jnp.sqrt(N_arr)
            else:
                L = jsp.linalg.cholesky(N_arr, lower=True)
                return key, L @ jax.random.normal(subkey, (N_arr.shape[0],))

        sample.params = list(N_fn.params) if N_fn is not None else []
        return sample

    @property
    def make_solve(self):
        return noisesolve(None, self.N)

    @property
    def make_whiten(self):
        # diag(N)^{-1/2} applicator for the timing-model projection.
        return dwhiten(None, self.N)

    @property
    def make_inv(self):
        if getattr(self, 'inv', None):
            # shortcut for GPs with custom make_inv
            return noiseinvfunc(self.inv)
        else:
            return noiseinv(self.N)

    def make_kernelproduct(self, y):
        return normal(y, self.make_solve)


class NoiseMatrix1D(NoiseMatrix):
    """Marker subclass for 1D (diagonal) noise. Same impl as NoiseMatrix.

    Exists so that `isinstance(x, matrix.NoiseMatrix1D_var)` dimension dispatch
    in likelihood.py still discriminates 1D vs 2D when the monkeypatch is active.
    """


class NoiseMatrix2D(NoiseMatrix):
    """Marker subclass for 2D (full) noise."""


def NoiseMatrix12D(getN):
    """Dispatch on getN.type, matching matrix.NoiseMatrix12D_var."""
    is_2d = getattr(getN, "type", None) is jax.Array
    return (NoiseMatrix2D if is_2d else NoiseMatrix1D)(getN)


class WoodburyKernel(Kernel):
    def __init__(self, N, F, P):
        self.N, self.F, self.P = N, F, P

    def make_sample(self):
        """Sample y ~ N(0, N + F P F^T) as noise + F @ coefficients."""
        N_sample = self.N.make_sample()
        P_sample = self.P.make_sample()
        F_const, F_fn = _materialize(self.F)

        F_params = list(F_fn.params) if F_fn is not None else []

        def sample(key, params={}):
            key, noise = N_sample(key, params)
            key, c = P_sample(key, params)
            F_arr = F_const if F_fn is None else F_fn(params=params)
            return key, noise + F_arr @ c

        sample.params = sorted(set(N_sample.params + P_sample.params + F_params))
        return sample

    @property
    def make_solve(self):
        return mm.prune_graph(woodbury(None, self.N.make_solve, self.F, self.P.make_inv), output='solve')

    def make_kernelproduct(self, y):
        return woodbury(y, self.N.make_solve, self.F, self.P.make_inv)

    def make_kernelsolve(self, y, T):
        """Callable returning (T^T Σ^-1 y, T^T Σ^-1 T) for Σ = N + F P F^T.

        Matches the contract of matrix.WoodburyKernel_var*.make_kernelsolve
        so likelihood.py's GlobalLikelihood.conditional path works unchanged.
        """
        graph = woodburykernelsolve(y, T, self.N.make_solve, self.F, self.P.make_inv)
        f = mm.func(graph)
        def call(params={}):
            return f(params=params)
        call.params = f.params
        return call

    def make_conditional(self, y):
        return mm.prune_graph(woodbury(y, self.N.make_solve, self.F, self.P.make_inv), output='cond')

    def make_coefficientproduct(self, y):
        cvars = list(self.index.keys())
        def getc(params):
            return jnp.concatenate([params[cvar] for cvar in cvars])
        getc.params = cvars

        return woodburylatent(y, self.N.make_solve, self.F, self.P.make_solve, getc)


class WoodburyProjKernel(Kernel):
    """Like ``WoodburyKernel`` but the improper timing model `M` is handled by
    *projection* (the flat-prior / float32-safe path) instead of a huge-variance
    (1e40) Gaussian prior. The remaining GP `F` (e.g. ECORR) is kept as an
    ordinary Woodbury block. See `woodbury_proj` and docs/design/single_precision/adr/0004-timing-model-projection.md.

    Drop-in for the per-pulsar noise in the fused array path: its `make_solve`
    returns the timing-projected inverse operator, so when the array kernel
    applies it to the outer commongp/globalgp bases the projection propagates
    through the whole likelihood — no 1e40 block is ever formed.

      N : base white-noise kernel (must expose `make_whiten`).
      M : timing design matrix (n_toa, m_tm) — projected OUT.
      F : kept GP basis (e.g. ECORR exposure) and P its prior kernel.
    """
    def __init__(self, N, M, F, P):
        self.N, self.M, self.F, self.P = N, M, F, P

    @property
    def make_solve(self):
        return mm.prune_graph(
            woodbury_proj(None, self.N.make_whiten, self.M, self.F, self.P.make_inv),
            output='solve')

    def make_kernelproduct(self, y):
        return woodbury_proj(y, self.N.make_whiten, self.M, self.F, self.P.make_inv)

    def make_sample(self):
        # Sampling y under a flat (improper) timing prior is ill-defined; the
        # projection only enters the likelihood. Fall back to the proper-prior
        # Woodbury sample if a caller ever needs draws.
        raise NotImplementedError(
            "WoodburyProjKernel.make_sample: timing projection is likelihood-only; "
            "use a proper-prior WoodburyKernel to draw samples.")


class GlobalWoodburyKernel(Kernel):
    def __init__(self, Ns, Fs, P):
        self.Ns, self.Fs, self.P = Ns, Fs, P

    def make_kernelproduct(self, ys):
        if isinstance(self.Ns, (tuple, list)):
            # old path: list of per-pulsar noise kernels
            return globalwoodbury(ys, [N.make_solve for N in self.Ns], self.Fs, self.P.make_inv)
        elif hasattr(self.Ns, 'Ns'):
            # compound kernel: vectorized fused two-level path.
            # Reference+delta (single-precision Half B) opt-in: when
            # BOTH frozen references are present -- the inner Phi_ref,in on the
            # inner kernel (self.Ns.P_ref) and the outer Phi_ref,gw on this kernel
            # (self.P_ref) -- route to the refdelta twins. Absent -> today's graph,
            # byte-identical. The references are frozen f64 covariance leaves built
            # once at the ArrayLikelihood boundary; theta_ref never reaches here.
            P_ref_in = getattr(self.Ns, 'P_ref', None)
            P_ref_out = getattr(self, 'P_ref', None)
            if P_ref_in is not None and P_ref_out is not None:
                joint_graph = vectorwoodburyjointsolve_refdelta(
                    ys, self.Fs, [N.make_solve for N in self.Ns.Ns],
                    self.Ns.Fs, self.Ns.P.make_inv, P_ref_in.make_inv)
                # Two separate prunes: the reference graph has only constant
                # outputs (folds wholesale -> the outer reference solve folds), the
                # increment graph is live. Bundling them would keep the constant
                # reference solve live and scaling with n_psr (see rung-1 emit note).
                refconst_graph = mm.prune_graph(joint_graph, output='refconst')
                refincr_graph = mm.prune_graph(joint_graph, output='refincr')
                return globalwoodbury_fused_refdelta(
                    refconst_graph, refincr_graph, self.P.make_inv, P_ref_out.make_inv)
            joint_graph = vectorwoodburyjointsolve(
                ys, self.Fs, [N.make_solve for N in self.Ns.Ns],
                self.Ns.Fs, self.Ns.P.make_inv)
            proj_graph = mm.prune_graph(joint_graph, output='projected')
            return globalwoodbury_fused(proj_graph, self.P.make_inv)
        else:
            # single noise matrix
            return globalwoodbury(ys, self.Ns.make_solve, self.Fs, self.P.make_inv)


class VectorWoodburyKernel(Kernel):
    def __init__(self, Ns, Fs, P):
        self.Ns, self.Fs, self.P = Ns, Fs, P

    @property
    def make_solve(self):
        return vectorwoodburysolve([None] * len(self.Ns),
                                   [N.make_solve for N in self.Ns],
                                   self.Fs, self.P.make_inv)

    def make_kernelproduct(self, ys):
        # Reference+delta opt-in: a frozen inner Phi_ref leaf in
        # self.P_ref routes to the batched single-level refdelta twin (CURN/IRN,
        # no Hellings-Downs). Absent -> today's vectorwoodbury, byte-identical.
        P_ref = getattr(self, 'P_ref', None)
        if P_ref is not None:
            return vectorwoodbury_refdelta(ys, [N.make_solve for N in self.Ns],
                                           self.Fs, self.P.make_inv, P_ref.make_inv)
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

    def _coefficient_leaves(self, transform):
        """Shared coefficient/prior leaves for the cross and residual forms.

        Returns (prior_graph, _coeffs, means_leaf, has_reparams).

        - ``transform``: callable or list of ``rp(params, c) -> (c, ldL)``.
        - ``self.means``: callable ``params -> a0``; centers the GP prior.
        """
        if transform is None:
            reparams = []
        elif isinstance(transform, (list, tuple)):
            reparams = list(transform)
        else:
            reparams = [transform]

        # Prior on c_for_prior: either the uniform-Phi wrap of `P.make_inv` or,
        # for a mixed-Phi compound (commongp + HD globalgp), the per-GP-sum
        # graph supplied by `CompoundGP._build_mixed_logprior`. Both look the
        # same to the component graphs — a GraphLeaf taking c_for_prior. Reusing
        # it unchanged is what keeps the exact dense HD coefficient prior in the
        # residual form for free.
        if getattr(self, 'prior', None) is not None:
            prior_graph = self.prior
        else:
            prior_graph = gaussian_coefficient_logprior(None, self.P.make_inv)

        # Build the params -> (c, ldL) FuncLeaf that bundles fold + reparams.
        # cvarsall: list-per-pulsar of {coeff-name: slice}.
        cvarsall = (self.index if isinstance(self.index, list)
                    else [{par: sl} for par, sl in self.index.items()])
        cvar_params = sum([list(cvars) for cvars in cvarsall], [])

        def _fold(params):
            return jnp.array([jnp.concatenate([params[cvar] for cvar in cvars])
                              for cvars in cvarsall])

        def _coeffs(params):
            c = _fold(params)
            ldL = 0.0
            for rp in reparams:
                c, dldL = rp(params, c)
                ldL = ldL + dldL
            return (c, ldL)

        _coeffs.params = sorted(set(
            cvar_params + sum([list(rp.params) for rp in reparams], [])))

        means_leaf = self.means if getattr(self, 'means', None) is not None else 0.0

        return prior_graph, _coeffs, means_leaf, bool(reparams)

    def make_kernelproduct_gpcomponent(self, ys, transform=None, extsignals=None):
        """ArrayLikelihood.clogL path, CROSS form. Returns a metamatrix graph.

        Composes the GP-coefficient log-likelihood from `vectorgpcomponent`:

            xi --[reparams]--> c  --(prior on c - means)--  data sees c

        Leaves go in as graphs / FuncLeafs / arrays; folding decides what
        runs at trace time vs runtime. The output is a graph with named
        subgraphs 'logp' and 'staged'; the method prunes to whichever
        matches the reparam state.

        - ``extsignals``: list of ``ExtSignal`` (each with .coeffs, .Fs).
        """
        prior_graph, _coeffs, means_leaf, has_rp = self._coefficient_leaves(transform)

        graph = vectorgpcomponent(
            ys,
            [N.make_solve for N in self.Ns],
            list(self.Fs),
            prior_graph,
            _coeffs,
            means_leaf,
            [es.coeffs for es in (extsignals or [])],
            [list(es.Fs) for es in (extsignals or [])],
        )
        return mm.prune_graph(graph, output=('staged' if has_rp else 'logp'))

    def make_residualproduct(self, ys, transform=None, extsignals=None):
        """Residual-form twin of make_kernelproduct_gpcomponent. Same contract,
        no FtNmF. See vectorresidualcomponent."""
        prior_graph, _coeffs, means_leaf, has_rp = self._coefficient_leaves(transform)

        graph = vectorresidualcomponent(
            ys,
            [N.make_solve for N in self.Ns],
            list(self.Fs),
            prior_graph,
            _coeffs,
            means_leaf,
            [es.coeffs for es in (extsignals or [])],
            [list(es.Fs) for es in (extsignals or [])],
        )
        return mm.prune_graph(graph, output=('staged' if has_rp else 'logp'))


class CompoundGP:
    def __new__(cls, x):
        if not isinstance(x, (list, tuple)):
            # if this is a single GP, just return it
            return x
        else:
            return super().__new__(cls)
    @staticmethod
    def _is_vector(gp):
        # per-pulsar F: commongp exposes F as a tuple; globalgp carries Fs as a list.
        if hasattr(gp, 'F') and isinstance(gp.F, tuple):
            return True
        if hasattr(gp, 'Fs') and isinstance(gp.Fs, (list, tuple)):
            return True
        return False

    def __init__(self, gplist):
        self.gplist = gplist
        # Mixed-Phi compound: at least one gp's Phi is dense (NoiseMatrix2D,
        # e.g. HD globalgp) while another is per-pulsar diagonal
        # (NoiseMatrix1D, e.g. commongp). The Phi.N arrays can't be concat'd
        # into a single batched Phi.
        phi_kinds = {('2D' if isinstance(gp.Phi, NoiseMatrix2D) else '1D')
                     for gp in gplist}
        self._mixed_phi = len(phi_kinds) > 1

        # The per-GP coefficient log-prior (with Phi=None) is only for the
        # vector / decentered path — commongp/globalgp whose coefficients are
        # sampled (clogL), where the joint prior decomposes as a sum of per-GP
        # contributions. A mixed compound of *marginalized* GPs (e.g. a constant
        # timing GP + a constant fourier-variance GP in a single PulsarLikelihood)
        # has no `.index`; it instead needs a real combined Phi (see `Phi`).
        self._mixed_vector = (self._mixed_phi
                              and all(hasattr(gp, 'index') for gp in gplist)
                              and all(self._is_vector(gp) for gp in gplist))
        if self._mixed_vector:
            self.prior = self._build_mixed_logprior(gplist)
        if all(hasattr(gp, 'index') for gp in gplist):
            # vector commongp (per-pulsar F tuples) or commongp + globalgp
            # (the globalgp carries Fs as a list). Either way → list-of-dicts,
            # one per pulsar, matching matrix.VectorCompoundGP.
            if all(self._is_vector(gp) for gp in gplist):
                self.index = [dict(g) for g in
                              zip(*[gp.index.items() for gp in gplist])]
            else:
                # non-vector (single PulsarLikelihood): cumulative-offset flat dict
                index, cnt = {}, 0
                for gp in gplist:
                    for var, sli in gp.index.items():
                        width = sli.stop - sli.start
                        index[var] = slice(cnt, cnt + width)
                        cnt += width
                self.index = index
    def _concat(self, vecmats):
        return functools.reduce(lambda x, y: concat(x, y), vecmats)
    @property
    def F(self):
        # for VectorWoodburyKernel: each gp exposes either F (tuple of per-pulsar
        # arrays, commongp-style) or Fs (list, globalgp-style). Normalize both.
        def _per_psr(gp):
            return gp.F if hasattr(gp, 'F') else gp.Fs
        if all(isinstance(_per_psr(gp), (tuple, list)) for gp in self.gplist):
            return tuple(self._concat(Fs) for Fs in zip(*(_per_psr(gp) for gp in self.gplist)))
        else:
            return self._concat([_per_psr(gp) for gp in self.gplist])

    @staticmethod
    def _build_mixed_logprior(gplist):
        """Per-GP-sum log-prior graph for a mixed-Phi compound.

        Input  arg : c_for_prior of shape (npsr, k_total).
        Output     : scalar `Σ_g log p_g(c[:, g_slice])`.

        Per-GP contribution (matches matrix.VectorCompoundGP.priorfunc):
          - diagonal Phi (NoiseMatrix1D, per-pulsar):
              `-0.5 sum(c_g^2 / Phi_g) - 0.5 sum(log|Phi_g|)`
          - dense Phi (NoiseMatrix2D, e.g. HD globalgp):
              `-0.5 c_flat^T Phi^-1 c_flat - 0.5 logdet(Phi)`,
            where c_flat is c_g.reshape(-1) (row-major: per-pulsar order
            matches gp.index slice ordering).
        """
        b = mm.GraphBuilder()
        c_for_prior = b.leaf(None, name='c_for_prior')

        widths = [next(iter(gp.index.values())).stop -
                  next(iter(gp.index.values())).start
                  for gp in gplist]
        offsets = [0]
        for w in widths:
            offsets.append(offsets[-1] + w)

        def _slice_op(s, e):
            return lambda c: c[:, s:e]

        def _dense_contrib(c_, Phi_):
            cf = c_.reshape(-1)
            return (-0.5 * cf @ jnp.linalg.solve(Phi_, cf)
                    - 0.5 * jnp.linalg.slogdet(Phi_)[1])

        def _diag_contrib(c_, Phi_):
            return (-0.5 * jnp.sum(c_ * c_ / Phi_)
                    - 0.5 * jnp.sum(jnp.log(jnp.abs(Phi_))))

        contribs = []
        for i, gp in enumerate(gplist):
            s, e = offsets[i], offsets[i + 1]
            phi_n_leaf = b.leaf(gp.Phi.N, name=f'gp{i}_phiN')
            c_slice = b.node(_slice_op(s, e), [c_for_prior],
                             description=f'c_for_prior[:,{s}:{e}]')
            if isinstance(gp.Phi, NoiseMatrix2D):
                contrib = b.node(_dense_contrib, [c_slice, phi_n_leaf],
                                 description=f'gp{i} dense logprior')
            else:
                contrib = b.node(_diag_contrib, [c_slice, phi_n_leaf],
                                 description=f'gp{i} diag logprior')
            contribs.append(contrib)

        logpr = contribs[0]
        for c in contribs[1:]:
            logpr = logpr + c
        b.named(logpr, 'logpr')
        return b.graph

    def _mixed_dense_Phi(self):
        """Combined dense Phi for a mixed 1D/2D *marginalized* compound.

        Promotes each per-pulsar-diagonal (1D) block to a dense diagonal and
        block-diagonals everything into one dense covariance, mirroring
        matrix.CompoundGP's mixed-const / mixed-var handling. Handles constant
        (static arrays) and variable (callables) Phi blocks.
        """
        parts = [(_materialize(gp.Phi.N), isinstance(gp.Phi, NoiseMatrix2D))
                 for gp in self.gplist]

        if all(fn is None for (const, fn), _ in parts):
            blocks = [const if is2d else jnp.diag(const)
                      for (const, fn), is2d in parts]
            return NoiseMatrix2D(jsp.linalg.block_diag(*blocks))

        def getN(params):
            blocks = []
            for (const, fn), is2d in parts:
                v = const if fn is None else fn(params=params)
                blocks.append(v if is2d else jnp.diag(v))
            return jsp.linalg.block_diag(*blocks)
        getN.params = sorted(set().union(
            *[set(fn.params) for (const, fn), _ in parts if fn is not None]))
        getN.type = jax.Array
        return NoiseMatrix2D(getN)

    @property
    def Phi(self):
        if self._mixed_vector:
            # Vector / decentered path: no single combined Phi; the prior lives
            # on `self.prior` and is consumed via VectorWoodburyKernel's `prior`
            # branch. likelihood.py threads this None through as P and never
            # calls P.make_inv when self.prior is set.
            return None
        if self._mixed_phi:
            # mixed 1D/2D but marginalized (no sampled coefficients): build a
            # real combined dense Phi for the Woodbury.
            return self._mixed_dense_Phi()
        N = self._concat([gp.Phi.N for gp in self.gplist])
        nm = NoiseMatrix(N)

        # combined_inv only available when every gp supplies a Phi_inv;
        # otherwise fall through to the default noiseinv graph.
        if all(getattr(gp, 'Phi_inv', None) is not None for gp in self.gplist):
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

    logp = g.combine_logp_f64(ytNmy, g.dot(FtNmy, mu), [lP, lS, lN])  # Half A: f64 final combine

# yt Km y = yt Nm y - yt Nm F (Pinv + FtNmF)^-1 Ft Nm y
# Tt Km y = Tt Nm y - Tt Nm F (Pinv + FtNmF)^-1 Ft Nm y
# Tt Km T = Tt Nm T - Tt Nm F (Pinv + FtNmF)^-1 Ft Nm T
# quindi mi mancano TtNmy, TtNmF, TtNmT;
# il primo e l'ultimo si possono ottenere da allsolve(y, T), ma TtNmF?

@mm.graph
def noiseallsolve(graph, y, F, N):
    result = N.allsolve(y, F)
