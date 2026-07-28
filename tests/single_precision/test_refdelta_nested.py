"""Reference oracle for the nested reference+delta increment (Piece 2).

This is a *math* spec test: it implements the nested fused-path increment from
the nested reference+delta formulas (see docs/metamatrix_dev.md) in plain
numpy and checks it against a brute-force dense-Sigma likelihood. It locks the
formulas the eventual metamath kernel must reproduce.

Two pulsars, per-pulsar intrinsic red noise (inner GP, diagonal Phi) + a dense
cross-pulsar GP (outer, e.g. HD). White noise is fixed (folds). The increment is
ΔlogL = logL(θ) − logL(θ_ref) formed directly (never as a difference of two big
logLs). The note's mpmath check shows the increment is exact to ~1e-16; here we
compare against the f64 brute *difference*, whose own conditioning floor (~1e-9 at
|logL|~1e3) sets a loose tolerance that still catches any real algebra error
(missing term / wrong sign / ref-vs-current), which would be O(move) ~ 1e-2.
"""
import numpy as np
import pytest
from scipy.linalg import block_diag, cho_factor, cho_solve

Np, M_IN, M_OUT = 2, 4, 3
NTOA = [20, 25]


@pytest.fixture(scope="module")
def problem():
    rng = np.random.default_rng(1)
    P = [dict(N=np.exp(rng.standard_normal(ni)) * 1e-2,
              y=rng.standard_normal(ni),
              F_in=rng.standard_normal((ni, M_IN)),
              F_out=rng.standard_normal((ni, M_OUT))) for ni in NTOA]

    def consts(p):
        Ninv = 1.0 / p['N']
        return dict(a=p['y'] @ (Ninv * p['y']),
                    b_in=p['F_in'].T @ (Ninv * p['y']),
                    b_out=p['F_out'].T @ (Ninv * p['y']),
                    G_in=p['F_in'].T @ (Ninv[:, None] * p['F_in']),
                    G_out=p['F_out'].T @ (Ninv[:, None] * p['F_out']),
                    H=p['F_in'].T @ (Ninv[:, None] * p['F_out']))
    C = [consts(p) for p in P]

    # reference covariances (the frozen Phi_ref leaves; well-conditioned Phi_gw)
    phi_in_ref = [np.exp(0.5 * rng.standard_normal(M_IN)) for _ in range(Np)]
    A = rng.standard_normal((Np * M_OUT, Np * M_OUT))
    phi_gw_ref = A @ A.T / (Np * M_OUT) + 3 * np.eye(Np * M_OUT)
    return P, C, phi_in_ref, phi_gw_ref


def _parts(C, phi_in_list, phi_gw):
    I = []
    for i in range(Np):
        c = C[i]
        cf = cho_factor(np.diag(1.0 / phi_in_list[i]) + c['G_in'])
        muy, muF = cho_solve(cf, c['b_in']), cho_solve(cf, c['H'])
        I.append(dict(cf=cf, muy=muy, muF=muF,
                      btil=c['b_out'] - c['H'].T @ muy,
                      Gtil=c['G_out'] - c['H'].T @ muF))
    btil = np.concatenate([x['btil'] for x in I])
    Gtil = block_diag(*[x['Gtil'] for x in I])
    Pm = np.linalg.inv(phi_gw)                       # prior inverse the kernel is handed
    cf_out = cho_factor(Pm + Gtil)
    return dict(I=I, btil=btil, Gtil=Gtil, Pm=Pm, cf_out=cf_out,
                nu=cho_solve(cf_out, btil))


def increment(C, phi_in_ref, phi_gw_ref, phi_in, phi_gw):
    """ΔlogL via the nested-increment formulas (inner → projection → outer)."""
    R, X = _parts(C, phi_in_ref, phi_gw_ref), _parts(C, phi_in, phi_gw)
    dat, dbt, dGt, ldin = [], [], [], 0.0
    for i in range(Np):
        c = C[i]
        dphi = phi_in[i] - phi_in_ref[i]
        dD = -dphi / (phi_in[i] * phi_in_ref[i])     # = 1/phi - 1/phi_ref, no cancellation
        dmuy = -cho_solve(X['I'][i]['cf'], dD * R['I'][i]['muy'])
        dmuF = -cho_solve(X['I'][i]['cf'], dD[:, None] * R['I'][i]['muF'])
        dat.append(-c['b_in'] @ dmuy)
        dbt.append(-c['H'].T @ dmuy)
        dGt.append(-c['H'].T @ dmuF)
        S0 = np.eye(M_IN) + np.diag(phi_in_ref[i]) @ c['G_in']
        ldin += np.linalg.slogdet(np.eye(M_IN) + np.linalg.solve(S0, np.diag(dphi) @ c['G_in']))[1]
    dA = sum(dat)
    dbt = np.concatenate(dbt)
    dGt_b = block_diag(*dGt)
    dD_gw = -X['Pm'] @ (phi_gw - phi_gw_ref) @ R['Pm']    # Pm - Pm_ref, routed (no inv-diff)
    dK = dD_gw + dGt_b
    dQ = dbt @ X['nu'] + R['btil'] @ cho_solve(X['cf_out'], dbt - dK @ R['nu'])
    S0o = np.eye(Np * M_OUT) + phi_gw_ref @ R['Gtil']
    mid = phi_gw_ref @ dGt_b + (phi_gw - phi_gw_ref) @ X['Gtil']
    ldout = np.linalg.slogdet(np.eye(Np * M_OUT) + np.linalg.solve(S0o, mid))[1]
    return -0.5 * ((dA - dQ) + (ldin + ldout))


def brute_logL(P, phi_in_list, phi_gw):
    Nf = np.concatenate([p['N'] for p in P])
    Fin = block_diag(*[p['F_in'] for p in P])
    Fout = block_diag(*[p['F_out'] for p in P])
    Pin = block_diag(*[np.diag(phi_in_list[i]) for i in range(Np)])
    Sig = np.diag(Nf) + Fin @ Pin @ Fin.T + Fout @ phi_gw @ Fout.T
    y = np.concatenate([p['y'] for p in P])
    ntot = len(Nf)
    cf = cho_factor(Sig)                              # Sigma is SPD: Cholesky logdet/solve
    logdet = 2.0 * np.sum(np.log(np.diag(cf[0])))
    return -0.5 * (y @ cho_solve(cf, y) + logdet + ntot * np.log(2 * np.pi))


@pytest.mark.parametrize("move", ["gw_only", "in_only", "both", "tiny"])
def test_nested_increment_matches_brute(problem, move):
    P, C, phi_in_ref, phi_gw_ref = problem
    rng = np.random.default_rng({"gw_only": 2, "in_only": 3, "both": 4, "tiny": 5}[move])
    if move == "gw_only":
        phi_in, phi_gw = phi_in_ref, phi_gw_ref * 1.4
    elif move == "in_only":
        phi_in = [phi_in_ref[i] * np.exp(0.3 * rng.standard_normal(M_IN)) for i in range(Np)]
        phi_gw = phi_gw_ref
    elif move == "both":
        phi_in = [phi_in_ref[i] * np.exp(0.3 * rng.standard_normal(M_IN)) for i in range(Np)]
        phi_gw = phi_gw_ref * 1.4
    else:  # tiny
        phi_in = [phi_in_ref[i] * (1 + 1e-3 * rng.standard_normal(M_IN)) for i in range(Np)]
        phi_gw = phi_gw_ref * (1 + 1e-3)

    d_brute = brute_logL(P, phi_in, phi_gw) - brute_logL(P, phi_in_ref, phi_gw_ref)
    d_form = increment(C, phi_in_ref, phi_gw_ref, phi_in, phi_gw)
    # loose tol = brute-difference conditioning floor; a real algebra bug is O(move)~1e-2.
    assert abs(d_form - d_brute) < 1e-6, f"{move}: formula={d_form} brute={d_brute}"


def test_fused_decomposition_matches_brute(problem):
    """The fused decomposition (a − b̃ᵀν + logdets) reproduces the dense logL."""
    P, C, phi_in_ref, phi_gw_ref = problem
    X = _parts(C, phi_in_ref, phi_gw_ref)
    a = sum(C[i]['a'] - C[i]['b_in'] @ X['I'][i]['muy'] for i in range(Np))
    quad = a - X['btil'] @ X['nu']
    ldin = sum(np.linalg.slogdet(np.eye(M_IN) + np.diag(phi_in_ref[i]) @ C[i]['G_in'])[1]
               for i in range(Np))
    ldout = np.linalg.slogdet(np.eye(Np * M_OUT) + phi_gw_ref @ X['Gtil'])[1]
    logdetN = sum(np.sum(np.log(p['N'])) for p in P)
    ntot = sum(NTOA)
    logL = -0.5 * (quad + logdetN + ldin + ldout + ntot * np.log(2 * np.pi))
    assert abs(logL - brute_logL(P, phi_in_ref, phi_gw_ref)) < 1e-8
