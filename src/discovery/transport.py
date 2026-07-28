"""Graph-consistent transport: a free-standing reparam object.

`Transport` replaces the in-likelihood decentering closure. Like
`likelihood_metamath.py` it is a *boundary module*: it may call
`metamatrix.func` at CONSTRUCTION time to bake constants, and the object it
returns participates in graphs only through the established FuncLeaf/reparam
contract. Kernel methods in `metamath.py` stay graph-pure.

Per pulsar, given blocks b = 1..B with bases W_b (n_toa, k_b) and live
conditioner precisions p_b(params) (k_b,), a frozen reference noise N0, and an
optional reference residual r0:

    W  = [W_1 | ... | W_B]                        (n_toa, k)
    G0 = W^T N0^-1 W                              baked once
    b0 = W^T N0^-1 r0                             baked once (centering)
    A(params)  = G0 + diag(p(params))            p = concat of p_b
    cf         = cho_factor(A, lower=True)        A = L L^T
    q          = mu + L^-T xi
    mu         = A^-1 b0                          0 if center=False
    ldJ        = -sum_i log L_ii                  log|dq/dxi|

mu is a translation (dmu/dxi = 0): it never enters ldJ. For any invertible
A(params) the map is a bijection with tracked Jacobian, so the transformed
density is exact regardless of how well A approximates the local posterior
precision. Failure semantics: construction raises, `validate`
raises, runtime `apply` is NaN-propagating JAX.
"""
import numpy as np
from dataclasses import dataclass

from . import metamatrix
from . import metamath
from . import utils as kh
from . import _kernels


@dataclass(frozen=True)
class TransportBlock:
    """One basis block of a transport. Construct via the adapters below."""
    name: str            # display name, e.g. 'rednoise'
    F: object            # (n_toa, k_b) ndarray -- constant basis at reference
    index: dict          # ONE coefficient-name -> slice(0, k_b) (localized)
    conditioner_precision: object   # callable params -> (k_b,); carries .params
    conditioner_kind: str = "exact_diagonal"  # or curn_inverse_marginal


def _validate_columns(name, F):
    F = np.asarray(F, dtype=np.float64)
    if F.ndim != 2 or F.shape[1] == 0:
        raise ValueError(f"transport block '{name}': basis must be a 2-D "
                         f"array with at least one column, got shape {F.shape}")
    norms = np.sqrt((F ** 2).sum(axis=0))
    bad = np.flatnonzero(~np.isfinite(norms) | (norms == 0.0))
    if bad.size:
        raise ValueError(
            f"transport block '{name}' has {bad.size} zero or non-finite "
            f"column(s) at indices {bad.tolist()}; every transport column "
            f"must be a finite, nonzero basis vector.")
    return F


def _eval_basis(F):
    """Materialize a constant basis. Arrays pass through; a graph-valued basis
    (dict) is materialized iff it is parameter-free -- mirroring the closure's
    `_eval_F` but refusing live bases."""
    if isinstance(F, dict):
        live = metamatrix.graph_params(F)
        if live:
            raise ValueError(
                f"transport bases must be constant; this basis graph depends "
                f"on parameters {live}. Materialize it at reference values "
                f"before building the transport.")
        return np.asarray(metamatrix.func(F)(params={}))
    return np.asarray(F)


def _single_key_index(index, psr_slot, width, blockname):
    """Select THE ONE coefficient key belonging to `psr_slot` from a factory
    index and localize it to slice(0, width).

    Factory index shapes: per-pulsar GP -> one-entry dict (psr_slot must
    be None); commongp/globalgp -> flat dict with one entry per pulsar in
    pulsar order, slices k*i..k*(i+1); CompoundGP -> list of dicts.
    """
    if isinstance(index, list):                       # CompoundGP form
        if psr_slot is None or psr_slot < 0 or psr_slot >= len(index):
            raise ValueError(
                f"transport block '{blockname}': a list-form index requires "
                f"a valid psr_slot; got {psr_slot!r} for {len(index)} entries.")
        index = index[psr_slot]
        items = list(index.items())
        if len(items) != 1:
            raise ValueError(
                f"transport block '{blockname}': selected pulsar index has "
                f"{len(items)} keys; adapters accept one factory GP per block.")
        key, sli = items[0]
        if (sli.stop - sli.start) != width:
            raise ValueError(
                f"transport block '{blockname}': coefficient slice for "
                f"'{key}' has width {sli.stop - sli.start}, but the basis has "
                f"{width} columns.")
        return {key: slice(0, width)}

    items = list(index.items())
    if psr_slot is None:
        if len(items) != 1:
            raise ValueError(
                f"transport block '{blockname}': index has {len(items)} keys; "
                f"pass psr_slot to select one pulsar's coefficients.")
        key, sli = items[0]
    else:
        if psr_slot < 0 or psr_slot >= len(items):
            raise ValueError(
                f"transport block '{blockname}': psr_slot={psr_slot} out of "
                f"range for an index with {len(items)} pulsar entries.")
        key, sli = items[psr_slot]
    if (sli.stop - sli.start) != width:
        raise ValueError(
            f"transport block '{blockname}': coefficient slice for "
            f"'{key}' has width {sli.stop - sli.start}, but the basis has "
            f"{width} columns.")
    return {key: slice(0, width)}


def gp_block(gp, psr_slot=None):
    """Adapt a discovery GP with a DIAGONAL prior to a TransportBlock.

    - per-pulsar GP (makegp_fourier on one pulsar): psr_slot=None;
      Phi.getN(params) has shape (k,).
    - commongp (makecommongp_fourier): psr_slot=i; gp.F is a tuple of
      per-pulsar bases and Phi.getN(params) has shape (npsr, k) -- the pulsar's
      ROW must be selected (exactly how the decenter closure consumes it before
      its axis-1 concat).
    """
    F = _eval_basis(gp.F[psr_slot] if psr_slot is not None else gp.F)
    F = _validate_columns(getattr(gp, 'gpname', 'gp'), F)
    getN = gp.Phi.getN

    if psr_slot is None:
        def conditioner_precision(params):
            return kh.jnp.asarray(getN(params)) ** -1
    else:
        def conditioner_precision(params, _i=psr_slot):
            return kh.jnp.asarray(getN(params))[_i] ** -1
    conditioner_precision.params = list(getattr(getN, 'params', []))

    name = getattr(gp, 'gpname', 'gp')
    return TransportBlock(name, F,
                          _single_key_index(gp.index, psr_slot, F.shape[1], name),
                          conditioner_precision)


def _conditioner_precision_from_spec(spec, k, name):
    """Normalize an `array_block` precision spec into a callable
    `params -> (k,)` carrying `.params`.

    Accepts a scalar (broadcast), a `(k,)` vector, or a callable with a
    `.params` attribute. Constant specs are positivity-checked eagerly (no
    floors); a live callable is trusted and validated at `validate()` time.
    """
    if callable(spec):
        params = list(getattr(spec, 'params', []))

        def cp(params_in, _f=spec):
            return kh.jnp.asarray(_f(params_in))
        cp.params = params
        return cp

    arr = np.asarray(spec, dtype=np.float64)
    if arr.ndim == 0:
        vec = np.full((k,), float(arr))
    elif arr.shape == (k,):
        vec = arr
    else:
        raise ValueError(
            f"array_block '{name}': conditioner_precision must be a scalar, a "
            f"({k},) vector, or a callable with .params; got shape {arr.shape}")
    bad = np.flatnonzero(~np.isfinite(vec) | (vec < 0.0))
    if bad.size:
        raise ValueError(
            f"array_block '{name}': constant conditioner_precision has "
            f"non-finite or negative entries at indices {bad.tolist()}; supply "
            f"a proper prior precision (no floors).")
    jvec = kh.jnparray(vec)

    def cp_const(params_in, _v=jvec):
        return _v
    cp_const.params = []
    return cp_const


def array_block(F, index, conditioner_precision, name="external"):
    """Caller-declared external transport block.

    `F` is a plain constant basis discovery does not interpret; `index` is a
    one-key `{name: slice(0, k)}` map naming the caller's coordinate; and
    `conditioner_precision` is MANDATORY -- the exact prior precision in the
    caller's sampling coordinate (a scalar, a `(k,)` vector, or a callable with
    `.params`). Discovery provides no default and no floor. Column
    validation is identical to the GP adapters.
    """
    F = _validate_columns(name, _eval_basis(F))
    k = int(F.shape[1])
    if not isinstance(index, dict) or len(index) != 1:
        raise ValueError(
            f"array_block '{name}': index must be a one-key dict "
            f"{{name: slice(0, k)}}; got {index!r}")
    (key, sli), = index.items()
    if not (isinstance(sli, slice) and (sli.start, sli.stop) == (0, k)):
        raise ValueError(
            f"array_block '{name}': index slice for '{key}' must be "
            f"slice(0, {k}); got {sli!r}")
    cp = _conditioner_precision_from_spec(conditioner_precision, k, name)
    return TransportBlock(name, F, {key: slice(0, k)}, cp)


def globalgp_curn_block(globalgp, psr_slot, npsr):
    """Per-pulsar conditioner view of a DENSE global GP: elementwise
    reciprocal of the dense Phi diagonal, reshaped per pulsar -- the existing
    decenter convention, kept for parity.

    INVERSE MARGINAL VARIANCE, NOT DENSE PRIOR PRECISION.
    The exact dense prior stays in the likelihood; this view exists only inside
    the transport, and the function name says so at every call site.
    """
    if npsr != len(globalgp.Fs) or npsr != len(globalgp.index):
        raise ValueError(
            f"globalgp_curn_block: npsr={npsr}, but the global GP has "
            f"{len(globalgp.Fs)} bases and {len(globalgp.index)} index entries")
    if psr_slot < 0 or psr_slot >= npsr:
        raise ValueError(
            f"globalgp_curn_block: psr_slot={psr_slot} out of range "
            f"for npsr={npsr}")
    F = _validate_columns(getattr(globalgp, 'gpname', 'gw'),
                          _eval_basis(globalgp.Fs[psr_slot]))
    getN = globalgp.Phi.getN

    def conditioner_precision(params, _i=psr_slot, _n=npsr):
        return kh.jnp.diag(getN(params) ** -1).reshape((_n, -1))[_i]
    conditioner_precision.params = list(getattr(getN, 'params', []))

    name = getattr(globalgp, 'gpname', 'gw')
    return TransportBlock(name, F,
                          _single_key_index(globalgp.index, psr_slot,
                                            F.shape[1], name),
                          conditioner_precision,
                          conditioner_kind="curn_inverse_marginal")


# --------------------------------------------------------------------------
# reference noise
# --------------------------------------------------------------------------

class _FrozenSolve:
    """One-method reference-noise operator: solve(rhs) -> (N0^-1 rhs, logdet).
    Built at construction; contains no free parameters by construction.

    ``diagonal`` is the exact diagonal of the reference covariance ``N0`` in
    canonical TOA order, retained so the geometry certifier can standardize a
    per-TOA residual remainder. It is stored, never reconstructed
    from live noise parameters."""
    def __init__(self, solve_fn, description, diagonal=None):
        self._solve, self.description = solve_fn, description
        self._diagonal = None if diagonal is None else np.asarray(
            diagonal, dtype=np.float64)

    def solve(self, rhs):
        return self._solve(rhs)

    def diagonal(self):
        """Return the exact ``diag(N0)`` in canonical TOA order."""
        if self._diagonal is None:
            raise NotImplementedError(
                f"reference noise '{self.description}' was built without an "
                f"exact diagonal; diag(N0) is unavailable")
        return self._diagonal


def _frozen_kernel_diagonal(kernel, params0):
    """Exact ``diag(N0)`` for a frozen metamath reference kernel.

    Supports the two families used as timing references: a plain diagonal/dense
    ``NoiseMatrix`` and the Sherman-Morrison ECORR ``NoiseMatrixSM``
    (``diag(N) + F P F^T`` with a 0/1 exposure ``F``, so the diagonal adds
    ``F P`` epoch-wise). Any other kernel raises ``NotImplementedError`` — the
    type is checked before touching ``kernel.N``, so composite kernels do not
    fall into ``_materialize``.
    """
    if not isinstance(kernel, (metamath.NoiseMatrix, metamath.NoiseMatrixSM)):
        raise NotImplementedError(
            f"exact diag(N0) is not implemented for reference kernel "
            f"{type(kernel).__name__}; supply reference_noise built from a "
            f"NoiseMatrix or NoiseMatrixSM")
    Nc, Nf = metamath._materialize(kernel.N)
    diag = np.asarray(Nc if Nf is None else Nf(params=params0), dtype=np.float64)
    if diag.ndim == 2:
        diag = np.diagonal(diag).astype(np.float64)
    if isinstance(kernel, metamath.NoiseMatrixSM):
        Pc, Pf = metamath._materialize(kernel.P)
        P = np.asarray(Pc if Pf is None else Pf(params=params0), dtype=np.float64)
        F = np.asarray(kernel.F, dtype=np.float64)
        diag = diag + (F * F) @ P
    return diag


def _live_kernel_diagonal(kernel, params):
    """Exact ``diag(C(params))`` for a LIVE marginalized metamath kernel.

    The eta-dependent analogue of :func:`_frozen_kernel_diagonal`: recurse through
    the folded Woodbury stack of a marginalized ``PulsarLikelihood.N`` at the
    current ``params``. Used only by
    :meth:`MarginalTransport.live_kernel_standard_deviation` for the decentered-
    mode geometry certifier; never inside NUTS. Never rebuilds ``F``/``P`` from
    signal factories or the pulsar -- reads only the kernel's own attributes.
    """
    # NoiseMatrixSM (ECORR) leaf: same base as the frozen path, but LIVE params.
    if isinstance(kernel, metamath.NoiseMatrixSM):
        Nc, Nf = metamath._materialize(kernel.N)
        diag = np.asarray(Nc if Nf is None else Nf(params=params), dtype=np.float64)
        if diag.ndim == 2:
            diag = np.diagonal(diag).astype(np.float64)
        Pc, Pf = metamath._materialize(kernel.P)
        P = np.asarray(Pc if Pf is None else Pf(params=params), dtype=np.float64)
        F = np.asarray(kernel.F, dtype=np.float64)
        return diag + (F * F) @ P
    # Projection / float32 path: out of scope for v1.
    if isinstance(kernel, metamath.WoodburyProjKernel):
        raise NotImplementedError(
            "_live_kernel_diagonal: WoodburyProjKernel (projection / float32 "
            "path) is not supported in v1; the float64 marginalized "
            "psl.N uses the 1e40 improper GP via WoodburyKernel.")
    # C = N_inner + F P F^T. Recurse on .N only; add diag(F P F^T).
    if isinstance(kernel, metamath.WoodburyKernel):
        base = _live_kernel_diagonal(kernel.N, params)
        F = np.asarray(kernel.F, dtype=np.float64)
        # diag(P(params)) from the prior kernel's own array (NoiseMatrix family).
        # A 1e40-scale improper prior contributes 1e40*(M_f (X) M_f)@1 -- FLOAT64
        # ONLY; it overflows float32 (use the projection path there).
        prior_arr = getattr(kernel.P, "N", None)
        if prior_arr is None:
            raise TypeError(
                f"_live_kernel_diagonal: WoodburyKernel prior "
                f"{type(kernel.P).__name__} exposes no .N array; a metamath "
                f"NoiseMatrix-family prior is required")
        Pc, Pf = metamath._materialize(prior_arr)
        Pmat = np.asarray(Pc if Pf is None else Pf(params=params), dtype=np.float64)
        if Pmat.ndim == 1:
            contrib = (F * F) @ Pmat                          # diagonal prior
        else:
            contrib = np.einsum("ij,jk,ik->i", F, Pmat, F)    # dense prior row form
        return base + contrib
    # Plain diagonal / dense noise leaf (NoiseMatrix, incl. NoiseMatrix1D).
    if isinstance(kernel, metamath.NoiseMatrix):
        Nc, Nf = metamath._materialize(kernel.N)
        diag = np.asarray(Nc if Nf is None else Nf(params=params), dtype=np.float64)
        if diag.ndim == 2:
            diag = np.diagonal(diag).astype(np.float64)
        return diag
    raise TypeError(
        f"_live_kernel_diagonal: unsupported kernel type "
        f"{type(kernel).__name__}; requires a metamath Woodbury stack "
        f"(NoiseMatrix / NoiseMatrixSM / WoodburyKernel).")


def reference_noise(psr):
    """Diagonal TOA-error reference: N0 = diag(toaerrs**2). EFAC=1, no
    EQUAD/ECORR. The dependency-free default."""
    n0 = np.asarray(psr.toaerrs, dtype=np.float64) ** 2
    kernel = metamath.NoiseMatrix(kh.jnparray(n0))
    f = metamatrix.func(kernel.make_solve)
    return _FrozenSolve(lambda rhs: f(rhs, params={}),
                        f"toaerrs diagonal ({psr.name})",
                        diagonal=n0)


def reference_noise_frozen(kernel, params0, description=None):
    """Freeze ANY metamath kernel's solve at explicit reference parameters
    params0 (e.g. a noisedict): works for measurement noise incl.
    Sherman-Morrison ECORR. This is the ONLY sanctioned way to use a
    parameterized kernel in a transport -- never call a live kernel with
    params={}."""
    params0 = dict(params0)  # freeze a snapshot; caller mutation cannot alter N0
    make_solve = getattr(kernel, 'make_solve', None)
    if make_solve is None or not isinstance(make_solve, dict):
        raise TypeError(
            f"reference_noise_frozen: expected a metamath kernel exposing "
            f"make_solve as a graph, got {type(kernel).__name__}. "
            f"(Legacy matrix.py kernels are not supported; build the model "
            f"under discovery.config(kernels='metamath').)")
    f = metamatrix.func(make_solve)
    missing = [p for p in f.params if p not in params0]
    if missing:
        raise ValueError(f"reference_noise_frozen: params0 is missing "
                         f"{missing}; a frozen reference must pin every "
                         f"parameter of the kernel it freezes.")
    try:
        diagonal = _frozen_kernel_diagonal(kernel, params0)
    except NotImplementedError:
        diagonal = None
    return _FrozenSolve(lambda rhs: f(rhs, params=params0),
                        description or f"frozen kernel at {sorted(params0)}",
                        diagonal=diagonal)


# --------------------------------------------------------------------------
# the Transport object
# --------------------------------------------------------------------------

_KINDS = ("exact_diagonal", "curn_inverse_marginal")


class Transport:
    """Per-pulsar frozen-reference transport  xi -> (q, ldJ).

    q = mu(params) + L(params)^-T xi,
    A(params) = G0 + diag(conditioner_precision(params)),  A = L L^T,
    ldJ = -sum(log diag L).
    """

    def __init__(self, blocks, *, reference_noise, reference_residual=None,
                 center=True, center_extsignals=None, psr_slot=None,
                 softclip=None):
        _kernels.require_metamath("Transport")

        blocks = list(blocks)
        if not blocks:
            raise ValueError("Transport requires at least one block")

        ntoas = {np.asarray(b.F).shape[0] for b in blocks}
        if len(ntoas) != 1:
            raise ValueError(f"transport blocks disagree on n_toa: "
                             f"{sorted(ntoas)}")
        self._ntoa = ntoas.pop()

        # -- index assembly: contiguous, non-overlapping, collision-checked --
        self.blocks = blocks
        offset, self.index = 0, {}
        self._block_slice = {}       # block name -> assembled q-slice (softclip)
        for b in self.blocks:
            if b.conditioner_kind not in _KINDS:
                raise ValueError(
                    f"block '{b.name}': unknown conditioner_kind "
                    f"{b.conditioner_kind!r}")
            if len(b.index) != 1:
                raise ValueError(
                    f"block '{b.name}': index must contain exactly one "
                    f"coefficient key; got {len(b.index)}")
            for key, sli in b.index.items():
                if key in self.index:
                    raise ValueError(f"duplicate coefficient key '{key}' "
                                     f"across transport blocks")
                if (sli.start, sli.stop) != (0, np.asarray(b.F).shape[1]):
                    raise ValueError(f"block '{b.name}': index slice must be "
                                     f"localized to slice(0, k_b); got {sli}")
                self.index[key] = slice(offset + sli.start, offset + sli.stop)
                if b.name in self._block_slice:
                    raise ValueError(
                        f"duplicate block name '{b.name}'; softclip and "
                        f"diagnostics key blocks by name, so names must be "
                        f"unique")
                self._block_slice[b.name] = self.index[key]
            offset += np.asarray(b.F).shape[1]
        self.dimension = offset

        # -- bake (construction-time boundary work) ---------------------------
        W = np.concatenate([np.asarray(b.F) for b in self.blocks], axis=1)
        self._W = kh.jnparray(W)             # retained for diagnostics/inverse
        N0mW, _ = reference_noise.solve(self._W)
        if not bool(np.all(np.isfinite(np.asarray(N0mW)))):
            raise ValueError(f"reference-noise solve produced non-finite "
                             f"values ({reference_noise.description})")
        self._G0 = kh.jnparray(W.T @ np.asarray(N0mW))
        self.reference_description = reference_noise.description
        # Retain the frozen reference-noise operator so the geometry certifier
        # can form N0^-1 quadratics and diag(N0) without reconstructing white/
        # ECORR noise from a notebook dictionary.
        self._reference_noise = reference_noise

        self._b0 = None
        if center:
            if reference_residual is None:
                raise ValueError("center=True requires reference_residual")
            r0 = np.asarray(reference_residual, dtype=np.float64)
            if r0.shape != (self._ntoa,):
                raise ValueError(f"reference_residual has shape {r0.shape}; "
                                 f"expected ({self._ntoa},)")
            self._b0 = kh.jnparray(np.asarray(N0mW).T @ r0)

        # -- ExtSignal-subtracted centering ----------------------------------
        # Bake E0_e = W^T N0^-1 Fext_i once; per eval subtract E0_e @ coeffs_e[i]
        # from b0 before the centering solve. A translation only; ldJ unchanged.
        self._extsignals = []
        ext_params = []
        if center_extsignals:
            if not center:
                raise ValueError(
                    "center_extsignals requires center=True (it moves the "
                    "centering translation)")
            N0mW_np = np.asarray(N0mW)
            for es in center_extsignals:
                esname = getattr(es, 'name', 'extsignal')
                Fs = getattr(es, 'Fs', None)
                coeffs = getattr(es, 'coeffs', None)
                if Fs is None or coeffs is None:
                    raise TypeError(
                        f"center_extsignals entry '{esname}' must expose .Fs "
                        f"and .coeffs (a discovery ExtSignal)")
                if psr_slot is None:
                    if len(Fs) != 1:
                        raise ValueError(
                            f"center_extsignals '{esname}': psr_slot is "
                            f"required to select one of {len(Fs)} pulsar bases")
                    slot = 0
                else:
                    slot = psr_slot
                if slot < 0 or slot >= len(Fs):
                    raise ValueError(
                        f"center_extsignals '{esname}': psr_slot={slot} out of "
                        f"range for {len(Fs)} pulsar bases")
                Fext = np.asarray(Fs[slot], dtype=np.float64)
                if Fext.shape[0] != self._ntoa:
                    raise ValueError(
                        f"center_extsignals '{esname}': Fs[{slot}] has "
                        f"{Fext.shape[0]} rows; expected n_toa={self._ntoa}")
                E0 = kh.jnparray(N0mW_np.T @ Fext)          # (k, k_ext)
                self._extsignals.append((E0, coeffs, slot, esname))
                ext_params += list(getattr(coeffs, 'params', []))

        # -- soft-clamp on named blocks' centering slices --------------------
        self._softclip = []
        if softclip:
            if not center:
                raise ValueError(
                    "softclip requires center=True; it clamps the centering "
                    "translation, which does not exist without centering")
            for bname, zmax in dict(softclip).items():
                if bname not in self._block_slice:
                    raise ValueError(
                        f"softclip names unknown block {bname!r}; blocks: "
                        f"{sorted(self._block_slice)}")
                if not (float(zmax) > 0.0):
                    raise ValueError(
                        f"softclip[{bname!r}]={zmax!r} must be positive")
                self._softclip.append(
                    (bname, self._block_slice[bname], float(zmax)))

        self.center = center
        self.params = sorted(set(
            sum([list(b.conditioner_precision.params) for b in self.blocks], [])
            + ext_params))

    # -- per-evaluation map (plain JAX; composes as a reparam FuncLeaf) ------
    def _factor(self, params):
        pinv = kh.jnp.concatenate(
            [b.conditioner_precision(params) for b in self.blocks])
        i1, i2 = kh.jnp.diag_indices(self.dimension)
        return (kh.jsp.linalg.cho_factor(
                    self._G0.at[i1, i2].add(pinv), lower=True),
                pinv)

    def apply(self, params, xi):
        cf, _ = self._factor(params)
        q = kh.jsp.linalg.solve_triangular(cf[0], xi, trans=1, lower=cf[1])
        ldJ = -kh.jnp.sum(kh.jnp.log(kh.jnp.diag(cf[0])))
        if self._b0 is not None:
            rhs = self._b0
            for E0, coeffs, slot, _name in self._extsignals:
                rhs = rhs - E0 @ kh.jnp.asarray(coeffs(params))[slot]
            mu = kh.jsp.linalg.cho_solve(cf, rhs)
            for _name, sli, zmax in self._softclip:
                mu = mu.at[sli].set(zmax * kh.jnp.tanh(mu[sli] / zmax))
            q = q + mu
        return q, ldJ

    def reference_noise_quadratic(self, vector):
        """Return ``vector^T N0^-1 vector`` under the frozen reference noise.

        Used by the geometry certifier's global residual-remainder RMS.
        """
        vector = kh.jnp.asarray(vector)
        if vector.shape != (self._ntoa,):
            raise ValueError(
                f"reference_noise_quadratic expects shape ({self._ntoa},); "
                f"got {tuple(vector.shape)}")
        solved, _ = self._reference_noise.solve(vector)
        out = vector @ solved
        if not bool(kh.jnp.isfinite(out)):
            raise ValueError("reference_noise_quadratic produced non-finite "
                             "output")
        return out

    def reference_noise_standard_deviation(self):
        """Return ``sqrt(diag(N0))`` in canonical TOA order."""
        diag = np.asarray(self._reference_noise.diagonal(), dtype=np.float64)
        if diag.shape != (self._ntoa,):
            raise ValueError(
                f"reference noise diagonal has shape {diag.shape}; expected "
                f"({self._ntoa},)")
        if not bool(np.all(diag > 0.0)) or not bool(np.all(np.isfinite(diag))):
            raise ValueError("reference noise diagonal must be finite and "
                             "strictly positive")
        return diag ** 0.5

    def split(self, q):
        """{coefficient-key: q[slice]} view, in self.index order."""
        return {key: q[sli] for key, sli in self.index.items()}

    def as_reparam(self):
        """The metamath reparam contract: rp(params, c) -> (c_out, ldL), with
        TRUE .params."""
        def rp(params, c):
            return self.apply(params, c)
        rp.params = list(self.params)
        return rp

    def validate(self, params):
        """Eager, non-JIT positivity/PD check. Raises ValueError with
        per-block diagnostics; returns the diagnostics dict on success."""
        offset = 0
        for b in self.blocks:
            p = np.asarray(b.conditioner_precision(params))
            kb = np.asarray(b.F).shape[1]
            if p.shape != (kb,):
                raise ValueError(f"block '{b.name}': conditioner_precision "
                                 f"returned shape {p.shape}; expected ({kb},)")
            bad = np.flatnonzero(~np.isfinite(p) | (p < 0.0))
            if bad.size:
                raise ValueError(
                    f"block '{b.name}': conditioner_precision has "
                    f"non-finite or negative entries at indices "
                    f"{bad.tolist()} (block-local). No floor is applied; "
                    f"fix the prior or remove the block.")
            offset += kb
        cf, _ = self._factor(params)
        diag = np.asarray(kh.jnp.diag(cf[0]))
        if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
            raise ValueError(
                f"transport factorization is not positive definite at the "
                f"given params (min Cholesky diagonal "
                f"{np.nanmin(diag):.3e}). A conditioned direction has no "
                f"prior support or the basis is degenerate; no floor is "
                f"applied. Blocks: "
                f"{[(b.name, np.asarray(b.F).shape[1]) for b in self.blocks]}")
        return self.diagnostics(params)

    def diagnostics(self, params=None, noise_solve=None):
        """Structural report plus optional eager numerical diagnostics.

        ``noise_solve`` is test-scale only. For diagonal-prior blocks its
        eigenvalues are the transformed local target metric. If this Transport
        contains a CURN block they are explicitly a *conditioner metric*: the
        dense cross-pulsar target precision cannot be represented per pulsar.
        """
        out = {
            "blocks": [
                {"name": b.name, "k": int(np.asarray(b.F).shape[1]),
                 "params": list(b.conditioner_precision.params),
                 "keys": list(b.index), "conditioner_kind": b.conditioner_kind}
                for b in self.blocks
            ],
            "dimension": self.dimension,
            "center": self.center,
            "reference_noise": self.reference_description,
        }
        if self._extsignals:
            out["center_extsignals"] = [name for *_, name in self._extsignals]
        if self._softclip:
            out["softclip"] = {name: zmax for name, _sli, zmax in self._softclip}
        if params is None:
            if noise_solve is not None:
                raise ValueError("diagnostics(noise_solve=...) requires params")
            return out

        cf, pinv = self._factor(params)
        p = np.asarray(pinv)
        d = np.asarray(kh.jnp.diag(cf[0]))
        out.update(
            precision_min=float(np.min(p)),
            precision_max=float(np.max(p)),
            chol_diag_min=float(np.min(d)),
            chol_diag_max=float(np.max(d)),
        )

        if noise_solve is not None:
            NmW, _ = noise_solve(self._W)
            H = np.asarray(self._W).T @ np.asarray(NmW) + np.diag(p)
            L = np.tril(np.asarray(cf[0]))
            Linv = np.linalg.solve(L, np.eye(self.dimension))
            eig = np.linalg.eigvalsh(Linv @ H @ Linv.T)
            out.update(
                metric_kind=("conditioner" if any(
                    b.conditioner_kind == "curn_inverse_marginal"
                    for b in self.blocks)
                    else "local_target"),
                metric_eig_min=float(eig[0]),
                metric_eig_max=float(eig[-1]),
            )
        return out

    def fingerprint(self):
        """Stable structural digest of this transport (block names/dims/order,
        parameter dependencies, centering, reference-noise description).

        Digests only the structure returned by ``diagnostics()`` (no params),
        so it is independent of any particular hyperparameter draw. Consumers
        (e.g. the nltiming dynamic run manifest) persist this so a saved run's
        transport can be reconciled without serializing an opaque closure.
        """
        import hashlib
        import json

        payload = json.dumps(
            {"schema": "discovery-transport-v1", "structure": self.diagnostics()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# ArrayTransport
# --------------------------------------------------------------------------

class ArrayTransport:
    """Per-pulsar transports behind one array reparam.

    Requirements (validated):
      - at least one transport;
      - EQUAL dimension k across pulsars (the array coefficient contract is a
        rectangular (npsr, k) array, so ragged transports cannot pass through
        it; ragged support is a non-goal);
      - all-or-none centering (mixed centering would silently shift some
        pulsars' coordinates and not others').
    """

    def __init__(self, transports):
        transports = list(transports)
        if not transports:
            raise ValueError("ArrayTransport requires at least one Transport")
        dims = sorted({t.dimension for t in transports})
        if len(dims) != 1:
            raise ValueError(
                f"ArrayTransport requires equal per-pulsar dimension; got "
                f"{dims}. Ragged transports are not supported; "
                f"use equal GP component counts across pulsars.")
        centers = {t.center for t in transports}
        if len(centers) != 1:
            raise ValueError("ArrayTransport requires all-or-none centering")
        if any(getattr(t, '_extsignals', None) or getattr(t, '_softclip', None)
               for t in transports):
            raise ValueError(
                "ArrayTransport does not support per-pulsar ExtSignal centering "
                "or softclip; the ragged joint path loops per-pulsar Transport "
                "objects instead")

        self.transports = transports
        self.dimension = dims[0]
        self.npsr = len(transports)
        self.center = centers.pop()
        self.params = sorted(set().union(*[set(t.params) for t in transports]))

        # batched bake: (npsr, k, k) and (npsr, k). This is the SAME batched
        # arithmetic as the decenter closure: preserve its call conventions.
        self._G0 = kh.jnparray([t._G0 for t in self.transports])
        self._b0 = (kh.jnparray([t._b0 for t in self.transports])
                    if self.center else None)

    def _pinv(self, params):
        return kh.jnp.stack([
            kh.jnp.concatenate([b.conditioner_precision(params)
                                for b in t.blocks])
            for t in self.transports])                    # (npsr, k)

    def apply(self, params, c):
        # c: (npsr, k) -- the array coefficient contract.
        i1, i2 = kh.jnp.diag_indices(self.dimension, ndim=2)
        cf = kh.jsp.linalg.cho_factor(
            self._G0.at[:, i1, i2].add(self._pinv(params)), lower=True)
        am = kh.jsp.linalg.solve_triangular(cf[0], c, trans=1, lower=cf[1])
        ldJ = -kh.jnp.logdet(cf[0][:, i1, i2])            # summed, as today
        if self._b0 is not None:
            am = am + kh.jsp.linalg.cho_solve(cf, self._b0)
        return am, ldJ

    def as_reparam(self):
        dim, npsr = self.dimension, self.npsr

        def rp(params, c):
            if c.shape != (npsr, dim):                    # trace-time check
                raise ValueError(
                    f"ArrayTransport: coefficient array has shape {c.shape}; "
                    f"the transport was built for ({npsr}, {dim}). The "
                    f"likelihood's coefficient width and the transport's "
                    f"blocks disagree.")
            return self.apply(params, c)
        rp.params = list(self.params)
        return rp

    def validate(self, params):
        return [t.validate(params) for t in self.transports]

    def diagnostics(self, params=None, noise_solve=None):
        """Aggregate per-pulsar diagnostics.

        noise_solve is None or a sequence of exactly npsr live solve callables.
        """
        if noise_solve is None:
            solves = [None] * self.npsr
        else:
            solves = list(noise_solve)
            if len(solves) != self.npsr:
                raise ValueError(
                    f"ArrayTransport diagnostics expected {self.npsr} "
                    f"noise solves; got {len(solves)}")
        return {
            "per_pulsar": [
                t.diagnostics(params=params, noise_solve=s)
                for t, s in zip(self.transports, solves)
            ],
            "dimension": self.dimension,
            "npsr": self.npsr,
            "params": list(self.params),
        }

    def fingerprint(self):
        """Stable structural digest of the array transport (see
        :meth:`Transport.fingerprint`)."""
        import hashlib
        import json

        payload = json.dumps(
            {"schema": "discovery-array-transport-v1", "structure": self.diagnostics()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# MarginalTransport: live-kernel decentering of one external block
# --------------------------------------------------------------------------


class MarginalTransport:
    """Live-kernel decentering for ONE external block:  xi -> (z, ldJ).

    ``z = mu(params) + L(params)^-T xi``,
    ``A(params) = W^T C(params)^-1 W + diag(p(params))``,  ``A = L L^T``,
    ``mu = A^-1 W^T C^-1 y`` (center=True) or ``0``,  ``ldJ = -sum(log diag L)``.

    ``C(params)`` is the LIVE marginalized covariance of the supplied metamath
    kernel (white noise + folded GPs + timing marginal blocks): the eta-dependent
    posterior-metric whitening of the external block. Unlike the joint
    :class:`Transport` (frozen ``N0``, sampled GP coefficients), the coefficients
    are analytically integrated inside ``C`` and only the ``W`` block is sampled.

    For the nltiming timing use case, ``y`` MUST be
    ``TimingLinearization.transport_effective_residual(raw_residual)`` and
    ``block.F`` MUST be ``TimingLinearization.sampled_basis``. Both are fixed at
    context construction / expansion time and never depend on the sampled
    coordinate (D-INV). Do not pass a delay-modified CompoundDelay.
    """

    def __init__(self, kernel, y, block, *, center=True):
        _kernels.require_metamath("MarginalTransport")

        make_ks = getattr(kernel, "make_kernelsolve", None)
        if make_ks is None:
            raise TypeError(
                "marginal_transport: kernel must be a metamath kernel exposing "
                "make_kernelsolve (pass the assembled PulsarLikelihood.N); got "
                f"{type(kernel).__name__}.")
        if callable(y):                                                   # D-INV
            raise TypeError(
                "marginal_transport: y must be a fixed residual array, not a "
                "delay-modified callable (CompoundDelay); a coordinate-dependent "
                "centering would break the triangular Jacobian (D-INV).")
        y = np.asarray(y, dtype=np.float64)

        if len(block.index) != 1:
            raise ValueError(
                f"MarginalTransport: block must have exactly one coefficient key "
                f"(a single array_block); got {len(block.index)}")
        (key, sli), = block.index.items()
        W = np.asarray(block.F, dtype=np.float64)          # validated by array_block
        if (sli.start, sli.stop) != (0, W.shape[1]):
            raise ValueError(
                f"MarginalTransport block '{key}': index slice must be "
                f"slice(0, {W.shape[1]}); got {sli}")
        if y.shape != (W.shape[0],):
            raise ValueError(f"y has shape {y.shape}; expected ({W.shape[0]},)")

        self.blocks = [block]                              # Transport-compatible
        self.index = {key: slice(0, W.shape[1])}
        self.dimension = int(W.shape[1])
        self._ntoa = int(W.shape[0])
        self.center = bool(center)
        self._kernel = kernel
        self._W = kh.jnparray(W)
        self._y = kh.jnparray(y)

        self._ks = make_ks(self._y, self._W)               # (W^T C^-1 y, W^T C^-1 W)
        # Live solve for the residual quadratic (NOT frozen at a snapshot).
        self._live_solve = metamatrix.func(kernel.make_solve)
        self.params = sorted(
            set(self._ks.params) | set(block.conditioner_precision.params))

    # -- per-evaluation map -------------------------------------------------
    def _factor(self, params):
        b, G = self._ks(params)
        pinv = self.blocks[0].conditioner_precision(params)
        i1, i2 = kh.jnp.diag_indices(self.dimension)
        A = G.at[i1, i2].add(pinv)
        return kh.jsp.linalg.cho_factor(A, lower=True), b, pinv

    def apply(self, params, xi):
        cf, b, _ = self._factor(params)
        z = kh.jsp.linalg.solve_triangular(cf[0], xi, trans=1, lower=cf[1])
        ldJ = -kh.jnp.sum(kh.jnp.log(kh.jnp.diag(cf[0])))
        if self.center:
            z = z + kh.jsp.linalg.cho_solve(cf, b)
        return z, ldJ

    def split(self, z):
        """{coefficient-key: z[slice]} view, in self.index order."""
        return {key: z[sli] for key, sli in self.index.items()}

    def as_reparam(self):
        """The metamath reparam contract: rp(params, c) -> (c_out, ldL), with
        TRUE .params (parity with Transport)."""
        def rp(params, c):
            return self.apply(params, c)
        rp.params = list(self.params)
        return rp

    # -- live-kernel geometry hooks (offline certifier only) ----------------
    def live_kernel_quadratic(self, params, vector):
        """Return ``vector^T C(params)^-1 vector`` under the LIVE marginal kernel.

        The eta-dependent analogue of ``Transport.reference_noise_quadratic``,
        via the SAME kernel graph -- no dense assembly, no second
        make_kernelsolve, no hand-rolled Woodbury.
        """
        vector = kh.jnp.asarray(np.asarray(vector, dtype=np.float64))
        if vector.shape != (self._ntoa,):
            raise ValueError(
                f"live_kernel_quadratic expects shape ({self._ntoa},); "
                f"got {tuple(vector.shape)}")
        xinv, _ld = self._live_solve(vector, params=params)
        out = float(vector @ xinv)
        if not np.isfinite(out):
            raise ValueError("live_kernel_quadratic produced non-finite output")
        return out

    def live_kernel_standard_deviation(self, params):
        """Return ``sqrt(diag(C(params)))`` in canonical TOA order.

        The eta-dependent analogue of
        ``Transport.reference_noise_standard_deviation``, computed by the
        recursive :func:`_live_kernel_diagonal` walk of the kernel's own Woodbury
        stack.
        """
        diag = np.asarray(_live_kernel_diagonal(self._kernel, params),
                          dtype=np.float64)
        if diag.shape != (self._ntoa,):
            raise ValueError(
                f"live kernel diagonal has shape {diag.shape}; expected "
                f"({self._ntoa},)")
        if not bool(np.all(diag > 0.0)) or not bool(np.all(np.isfinite(diag))):
            raise ValueError("live kernel diagonal must be finite and strictly "
                             "positive")
        return diag ** 0.5

    # -- validation / diagnostics / fingerprint -----------------------------
    def validate(self, params):
        """Eager positivity/PD check (no floors). Mirrors Transport.validate."""
        b = self.blocks[0]
        p = np.asarray(b.conditioner_precision(params))
        kb = np.asarray(b.F).shape[1]
        if p.shape != (kb,):
            raise ValueError(f"block '{b.name}': conditioner_precision returned "
                             f"shape {p.shape}; expected ({kb},)")
        bad = np.flatnonzero(~np.isfinite(p) | (p < 0.0))
        if bad.size:
            raise ValueError(
                f"block '{b.name}': conditioner_precision has non-finite or "
                f"negative entries at indices {bad.tolist()}; no floor is "
                f"applied.")
        cf, _b, _pinv = self._factor(params)
        diag = np.asarray(kh.jnp.diag(cf[0]))
        if not np.all(np.isfinite(diag)) or np.any(diag <= 0.0):
            raise ValueError(
                f"marginal transport factorization is not positive definite at "
                f"the given params (min Cholesky diagonal {np.nanmin(diag):.3e}); "
                f"no floor is applied.")
        return self.diagnostics(params)

    def diagnostics(self, params=None, noise_solve=None):
        """Structural report plus optional eager Cholesky diagnostics.

        ``noise_solve`` is NOT supported: the metric is live, so the honest
        geometry check is the identity / ``certify_decentered_geometry``, not a
        frozen-noise substitute.
        """
        if noise_solve is not None:
            raise ValueError(
                "MarginalTransport.diagnostics does not support noise_solve "
                "(the metric is live; use certify_decentered_geometry)")
        b = self.blocks[0]
        out = {
            "blocks": [
                {"name": b.name, "k": int(np.asarray(b.F).shape[1]),
                 "params": list(b.conditioner_precision.params),
                 "keys": list(b.index), "conditioner_kind": b.conditioner_kind}
            ],
            "dimension": self.dimension,
            "center": self.center,
            "reference_noise": "live_kernel",
            "kernel_measurement": getattr(self._kernel, "measurement", None),
        }
        if params is None:
            return out
        cf, _b, pinv = self._factor(params)
        p = np.asarray(pinv)
        d = np.asarray(kh.jnp.diag(cf[0]))
        out.update(
            precision_min=float(np.min(p)),
            precision_max=float(np.max(p)),
            chol_diag_min=float(np.min(d)),
            chol_diag_max=float(np.max(d)),
        )
        return out

    def fingerprint(self):
        """Stable structural digest under schema ``discovery-marginal-transport-v1``."""
        import hashlib
        import json

        payload = json.dumps(
            {"schema": "discovery-marginal-transport-v1",
             "structure": self.diagnostics()},
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def marginal_transport(kernel, y, block, *, center=True):
    """Public factory for :class:`MarginalTransport`. See its docstring.

    ``kernel`` MUST be the assembled marginalized ``PulsarLikelihood.N`` (a
    metamath Woodbury *stack* exposing ``make_kernelsolve``) — the object whose
    ``C(params) = N0 + F Phi(params) F^T`` folds white noise together with the
    RN/DM and marginalized-timing GP priors. A bare white-noise leaf
    (``metamath.NoiseMatrix``) has no ``make_kernelsolve`` and is rejected with a
    ``TypeError``; do not "simplify" to the WN kernel — the eta-dependence of the
    transport lives entirely in the folded GP priors.
    """
    return MarginalTransport(kernel, y, block, center=center)
