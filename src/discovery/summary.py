"""Human-readable snapshots of a built model.

A discovery likelihood is assembled from a *list* of components --
``psr.residuals``, a white-noise kernel, and one or more Gaussian-process
signals (timing model, ECORR, red noise, a correlated GW process, ...). Once
assembled those components are folded into a single Woodbury kernel, so the
obvious handle, ``logL.params``, only reports the *free* parameters and hides
everything that was fixed (e.g. white noise pinned from a noise dictionary) or
that contributes a basis but no free parameter (the timing-model and ECORR
GPs). That makes it hard to see, at a glance, what model you actually built.

This module reconstructs that picture from the original components (which the
likelihood objects now retain) and renders it three ways, mirroring
enterprise's ``pta.summary()``:

* ``model.summary()``      -> a plain-text table (print it, or log it);
* ``model.summary_frame()``-> a pandas ``DataFrame``, one row per signal;
* ``model._repr_html_()``  -> a rich table when displayed in a notebook.

Everything here is pure introspection: it reads declarative attributes the
factories already set and never calls into the kernel math, so it works
unchanged under either the ``matrix`` or the ``metamath`` backend.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import prior


# parameter-name suffix encoding an array shape, e.g. 'rednoise_log10_rho(30)'
_SHAPE_RE = re.compile(r'\(([\d,\s]+)\)$')
# coefficient-vector name set by every GP factory: '{psr}_{signal}_coefficients(N)'
_COEFF_RE = re.compile(r'^(?P<psr>.+?)_(?P<sig>.+)_coefficients\((?P<n>\d+)\)$')


@dataclass
class SignalInfo:
    """One row of a model summary: a single signal in one signal collection."""
    name: str                              # signal label, e.g. 'rednoise', 'gw'
    kind: str                              # human description of what it is
    scope: str = 'per-pulsar'              # per-pulsar | common | global | data
    basis_shape: tuple | None = None       # (Ntoa, Nbasis) design-matrix shape
    varying: list = field(default_factory=list)   # free parameter names
    fixed: list = field(default_factory=list)      # pinned parameter names
    handle: str | None = None              # accessor, e.g. 'signals[4]' / 'commongp'
    slices: dict | None = None             # label -> slice, for stacked bases
    # How this block appears to the COEFFICIENT frontend (`clogL`); the marginal
    # frontend (`logL`) integrates every GP block analytically. Derived from the
    # assembled kernel's `.index` -- never from GP type. See
    # `_coefficients_label`.
    coefficients: str = '—'


# --------------------------------------------------------------------------
# describing individual components
# --------------------------------------------------------------------------

def _is_array(x):
    return isinstance(x, np.ndarray) or (hasattr(x, 'shape') and not hasattr(x, 'Phi')
                                         and not hasattr(x, 'params') and not callable(x))


def _signal_name(gp):
    """Best label for a GP: its ``gpname`` tag, else parsed from its index."""
    nm = getattr(gp, 'gpname', None)
    if nm:
        return nm
    idx = getattr(gp, 'index', None)
    if idx:
        m = _COEFF_RE.match(next(iter(idx)))
        if m:
            return m.group('sig')
    return type(gp).__name__


def _basis_shape(gp):
    F = getattr(gp, 'F', None)
    if F is not None and hasattr(F, 'shape'):
        return tuple(F.shape)
    Fs = getattr(gp, 'Fs', None)
    if Fs:
        F0 = Fs[0]
        if hasattr(F0, 'shape'):
            return tuple(F0.shape)
    return None


def _kernel_params(Phi):
    """Free parameter names of a prior kernel, across both backends: the
    matrix path exposes ``Phi.params`` directly; the metamath path keeps the
    parameterized callable at ``Phi.getN.params``."""
    pars = getattr(Phi, 'params', None)
    if pars:
        return list(pars)
    return list(getattr(getattr(Phi, 'getN', None), 'params', None) or [])


def _prior_params(gp):
    """Free parameter names a GP contributes (prior + optional mean)."""
    pars = list(_kernel_params(getattr(gp, 'Phi', None)))
    for mattr in ('mean', 'means'):
        mfn = getattr(gp, mattr, None)
        if mfn is not None:
            pars += list(getattr(mfn, 'params', []) or [])
    # de-duplicate, keep order
    seen, out = set(), []
    for p in pars:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _coefficients_label(gp, keys, assembled):
    """The `coefficients` column for one GP row.

    *keys* are the GP's own coefficient-index keys; *assembled* is the set of
    coefficient keys the applicable frontend actually samples, read off the
    assembled kernel's `.index`. Ground truth is that index, not the GP's type:
    `logL` marginalizes every GP, and a variable GP shadowed by
    `concat=False` is marginalized too.
    """
    if getattr(gp, 'project', False):
        # Projected out of the kernel entirely, never a coefficient.
        return 'projected'

    keys = list(keys or [])
    if not keys:
        return 'marginalized'

    present = [k for k in keys if k in assembled]
    if not present:
        return 'marginalized'
    if len(present) == len(keys):
        width = sum(_coeff_width(k) for k in keys)
        return f'sampled ({width})'

    missing = [k for k in keys if k not in assembled]
    raise ValueError(
        f"summary: GP '{_signal_name(gp)}' has some but not all of its "
        f"coefficient keys in the assembled index — present {present}, "
        f"missing {missing}. A GP block is either sampled whole or "
        f"marginalized whole; this is an internal consistency error.")


def _coeff_width(key):
    m = _COEFF_RE.match(key)
    return int(m.group('n')) if m else 0


def describe_component(obj, assembled=frozenset()):
    """Map one input component to a :class:`SignalInfo` (or ``None`` to skip).

    Recognizes: the residual vector (data), deterministic delays, the
    white-noise kernel, per-pulsar GPs (timing model, ECORR, Fourier red
    noise, ...). Correlated common/global GPs are handled by
    :func:`describe_global` because they fan out over pulsars.
    """
    # residual data vector
    if _is_array(obj):
        return SignalInfo('residuals', f'data ({np.shape(obj)[0]} TOAs)', scope='data')

    # white-noise kernel (tagged by makenoise_measurement*)
    meas = getattr(obj, 'measurement', None)
    if meas is not None:
        pars = list(meas['params']) + list(meas.get('ecorr_params', []))
        kind = 'white noise' + (' + ECORR (Sherman-Morrison)' if meas.get('ecorr') else '')
        info = SignalInfo(meas['name'], kind, basis_shape=meas.get('ecorr_basis_shape'),
                          coefficients='kernel')
        if meas['fixed']:
            info.fixed = pars
        else:
            info.varying = pars
        return info

    # a Gaussian-process signal: ConstantGP (fixed prior) or VariableGP
    if hasattr(obj, 'Phi') and (hasattr(obj, 'F') or hasattr(obj, 'Fs')):
        varying = _prior_params(obj)
        constant = type(obj).__name__ == 'ConstantGP' or not varying
        kind = 'GP, fixed prior' if constant else 'GP, variable prior'
        return SignalInfo(_signal_name(obj), kind, basis_shape=_basis_shape(obj),
                          varying=varying,
                          coefficients=_coefficients_label(
                              obj, getattr(obj, 'index', None) or {}, assembled))

    # deterministic delay (a plain callable carrying .params)
    if callable(obj):
        name = getattr(obj, 'name', None) or getattr(obj, '__name__', 'delay')
        return SignalInfo(name, 'deterministic delay',
                          varying=list(getattr(obj, 'params', []) or []))

    return None


def describe_global(gp, scope='global', assembled=frozenset()):
    """Describe a correlated common/global GP (or a compound of them).

    These fan a single process out over every pulsar, so identity is read from
    the per-pulsar coefficient-index keys: one summary row per distinct signal
    name found there.
    """
    if isinstance(gp, list):
        out = []
        for g in gp:
            out.extend(describe_global(g, scope, assembled))
        return out

    idx = getattr(gp, 'index', None) or {}
    # group index keys by signal name, count pulsars and basis width, and record
    # each pulsar's slice into the stacked coefficient vector.
    sigs = {}
    for key, sli in idx.items():
        m = _COEFF_RE.match(key)
        if not m:
            continue
        sig = m.group('sig')
        n = int(m.group('n'))
        entry = sigs.setdefault(sig, {'npsr': 0, 'n': n, 'slices': {}, 'keys': []})
        entry['npsr'] += 1
        entry['slices'][m.group('psr')] = sli
        entry['keys'].append(key)

    allpars = _prior_params(gp)            # prior params + any common mean params
    orfnames = getattr(gp, 'orfnames', None)

    out = []
    names = sorted(sigs, key=len, reverse=True)   # longest first for attribution
    claimed = set()
    for sig in names:
        meta = sigs[sig]
        # common/global parameter names embed the signal token, either bare
        # ('gw_log10_A') or pulsar-prefixed ('B1855+09_rednoise_gamma'); match
        # on the token surrounded by separators. Longest signal name claims first.
        token = f'_{sig}_'
        pars = [p for p in allpars
                if (token in f'_{p}_') and p not in claimed]
        claimed.update(pars)
        orf = ', '.join(orfnames) if (orfnames and len(sigs) == 1) else None
        kind = 'correlated GP' if scope == 'global' else 'common GP'
        if orf:
            kind += f' ({orf})'
        out.append(SignalInfo(
            sig, kind, scope=scope,
            basis_shape=(meta['npsr'], meta['n']),
            varying=pars, slices=meta['slices'],
            coefficients=_coefficients_label(gp, meta['keys'], assembled)))
    # restore index order (we sorted by length for attribution)
    out.sort(key=lambda s: list(sigs).index(s.name))
    # any leftover params (e.g. a combined-CRN's gw_* terms, or non-zero-mean
    # amplitudes) belong to this GP too — attach them rather than drop them.
    leftover = [p for p in allpars if p not in claimed]
    if leftover:
        if out:
            out[-1].varying = out[-1].varying + leftover
        else:
            out.append(SignalInfo(_signal_name(gp),
                                  'correlated GP' if scope == 'global' else 'common GP',
                                  scope=scope, varying=leftover,
                                  coefficients=_coefficients_label(gp, [], assembled)))
    return out


def describe_extsignal(ext):
    # an external signal carries a per-pulsar (Ntoa x n_ext) basis; the
    # informative number is n_ext, the count of its own basis terms.
    shape = _basis_shape(ext)
    nbasis = (shape[-1],) if shape else None
    return SignalInfo(getattr(ext, 'name', 'extsignal'), 'deterministic signal',
                      scope='external',
                      varying=list(getattr(ext, 'params', []) or []),
                      basis_shape=nbasis, coefficients='deterministic')


# --------------------------------------------------------------------------
# collecting a whole model into signal collections
# --------------------------------------------------------------------------

def _pulsar_signals(psl, assembled=frozenset()):
    """[SignalInfo] for one PulsarLikelihood, from its retained components.

    Each row carries its ``signals[i]`` handle — the i is the position in the
    pulsar's own ``signals`` list, so it is a live, zero-copy accessor."""
    sigs = []
    for i, obj in enumerate(getattr(psl, 'signals', [])):
        info = describe_component(obj, assembled)
        if info is not None and info.scope != 'data':
            info.handle = f'signals[{i}]'
            sigs.append(info)
    return sigs


def _psl_name(psl):
    """The pulsar's name. ``self.name`` is only set when a GP supplied it, so
    fall back to the GP `.name` tags or the white-noise kernel's psrname tag —
    important for white-noise-only models."""
    nm = getattr(psl, 'name', None)
    if nm:
        return nm
    for obj in getattr(psl, 'signals', []):
        n = getattr(obj, 'name', None)
        if n:
            return n
        meas = getattr(obj, 'measurement', None)
        if meas and meas.get('psrname'):
            return meas['psrname']
    return 'pulsar'


def _outer_coefficient_keys(model):
    """The coefficient keys an ArrayLikelihood's array frontend samples.

    Ground truth is the assembled kernel's `.index` — `_coefficient_assembly`
    on the metamath route, a list-of-dicts (one per pulsar) or a flat dict. The
    legacy matrix ArrayLikelihood has no such property; it merges the same
    per-GP `.index` dicts inside `clogL`, so fall back to that same merge. Both
    read an index, never a GP type.
    """
    assembly = getattr(model, '_coefficient_assembly', None)
    if assembly is not None:
        index = assembly[0].index
        if isinstance(index, list):
            return {key for d in index for key in d}
        return set(index or {})

    keys = set()
    commongp = getattr(model, 'commongp', None)
    gps = (commongp if isinstance(commongp, list) else [commongp]) if commongp is not None else []
    globalgp = getattr(model, 'globalgp', None)
    if globalgp is not None:
        gps = list(gps) + [globalgp]
    for g in gps:
        keys.update(getattr(g, 'index', None) or {})
    return keys


def _collect(model):
    """Return (collections, commons) where *collections* is a list of
    (label, ntoa, [SignalInfo]) per pulsar and *commons* is a list of
    SignalInfo for shared/correlated/global signals."""
    collections, commons = [], []

    commongp = getattr(model, 'commongp', None)
    globalgp = getattr(model, 'globalgp', None)
    psls = getattr(model, 'psls', None)

    # Route the assembled coefficient-key set to each row. Treatment is a
    # property of the FRONTEND, not of a GP: `logL` marginalizes everything,
    # `clogL` samples exactly what the assembled `.index` exposes.
    if psls is None:
        # single PulsarLikelihood: its own kernel's index is what clogL samples.
        per_psr = [set(getattr(getattr(model, 'N', None), 'index', None) or {})]
        outer = set()
    elif commongp is None and globalgp is None:
        # no outer assembly: ArrayLikelihood.clogL is the sum of psl.clogL, so
        # each row compares with that pulsar's own kernel index.
        per_psr = [set(getattr(getattr(psl, 'N', None), 'index', None) or {})
                   for psl in psls]
        outer = set()
    else:
        # an outer coefficient assembly exists: per-pulsar GPs inside each psl.N
        # belong to the inner noise solve and are analytically marginalized by
        # the array coefficient frontend, so their assembled key set is empty.
        per_psr = [set() for _ in psls]
        outer = _outer_coefficient_keys(model)

    if psls is None:                       # single PulsarLikelihood
        collections.append((_psl_name(model), _ntoa(model),
                            _pulsar_signals(model, per_psr[0])))
    else:
        for psl, assembled in zip(psls, per_psr):
            collections.append((_psl_name(psl), _ntoa(psl),
                                _pulsar_signals(psl, assembled)))

    if commongp is not None:
        cg = commongp if isinstance(commongp, list) else [commongp]
        for j, g in enumerate(cg):
            handle = f'commongp[{j}]' if isinstance(commongp, list) else 'commongp'
            for info in describe_global(g, scope='common', assembled=outer):
                info.handle = handle
                commons.append(info)

    if globalgp is not None:
        for info in describe_global(globalgp, scope='global', assembled=outer):
            info.handle = 'globalgp'
            commons.append(info)

    for k, ext in enumerate(getattr(model, 'extsignals', None) or []):
        info = describe_extsignal(ext)
        info.handle = f'extsignals[{k}]'
        commons.append(info)

    return collections, commons


def _ntoa(psl):
    # the residual vector is the reliable TOA count; when delays are present
    # psl.y is a CompoundDelay wrapper, so fall back to the residual component.
    y = getattr(psl, 'y', None)
    try:
        if isinstance(y, np.ndarray) or (hasattr(y, 'shape') and not hasattr(y, 'y')):
            return int(np.shape(y)[0])
    except Exception:
        pass
    for obj in getattr(psl, 'signals', []):
        if _is_array(obj):
            return int(np.shape(obj)[0])
    return None


def _prior_range(name):
    try:
        a, b = prior.getprior_uniform(name)
        return f'[{a:g}, {b:g}]'
    except Exception:
        return ''


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _backend_label():
    from . import utils
    import discovery
    kern = getattr(discovery, '_KERNELS', 'matrix')
    dtype = 'float64' if getattr(utils.jnp, 'float64', None) else 'float32'
    # x64 flag is the reliable signal for precision
    try:
        import jax
        prec = 'float64' if jax.config.read('jax_enable_x64') else 'float32'
    except Exception:
        prec = dtype
    return f'backend: {kern}, {prec}'


def _totals(collections, commons):
    varying, fixed = set(), set()
    basis = 0
    for _, _, sigs in collections:
        for s in sigs:
            varying.update(s.varying)
            fixed.update(s.fixed)
            if s.basis_shape and len(s.basis_shape) == 2 and s.scope != 'data':
                basis += s.basis_shape[1]
    for s in commons:
        varying.update(s.varying)
        fixed.update(s.fixed)
        # only marginalized GP bases count toward the GP problem size; a
        # deterministic external signal's (Ntoa x n_ext) basis is not a GP.
        if s.scope in ('common', 'global') and s.basis_shape and len(s.basis_shape) == 2:
            basis += s.basis_shape[0] * s.basis_shape[1]
    # "common" = a free parameter not prefixed by any single pulsar's name
    psrnames = [label for label, _, _ in collections]
    common = {p for p in varying
              if not any(p.startswith(nm + '_') for nm in psrnames)}
    return dict(varying=varying, fixed=fixed, common=common, basis=basis)


def summary(model, include_params=None, *, show_free=True, show_fixed=True,
            show_access=False, to_stdout=False):
    """Build (or print) a plain-text snapshot of *model*.

    Parameters
    ----------
    show_free, show_fixed : bool
        List each signal's free / fixed parameters (with uniform prior ranges
        for the free ones) beneath its row. Independent toggles.
    show_access : bool
        Add an ``access`` column with the live ``signals[i]`` / ``commongp`` /
        ``globalgp`` handle for each signal.
    include_params : bool, optional
        Back-compatible alias: ``True`` -> show both free and fixed, ``False``
        -> show neither. Overrides ``show_free`` / ``show_fixed`` when given.
    to_stdout : bool
        Print the summary instead of returning it.
    """
    if include_params is not None:
        show_free = show_fixed = bool(include_params)

    collections, commons = _collect(model)
    tot = _totals(collections, commons)

    W = 94 + (20 if show_access else 0)
    lines = []
    title = type(model).__name__
    lines.append(f'discovery model summary   ({_backend_label()})')
    lines.append('=' * W)

    def _shape(s):
        if not s.basis_shape:
            return '-'
        if len(s.basis_shape) == 2 and s.scope in ('global', 'common'):
            return f'{s.basis_shape[0]}psr x {s.basis_shape[1]}'
        return ' x '.join(str(d) for d in s.basis_shape)

    def _emit(sigs, header, handle_prefix=''):
        lines.append(header)
        lines.append('-' * W)
        head = (f'{"signal":<22}{"kind":<34}{"basis":<14}{"free":>6}'
                f'  {"coefficients":<14}')
        if show_access:
            head += f'  {"access":<20}'
        lines.append(head)
        lines.append('-' * W)            # F1: rule under the column header
        for s in sigs:
            row = (f'{s.name:<22}{s.kind:<34}{_shape(s):<14}{len(s.varying):>6}'
                   f'  {s.coefficients:<14}')
            if show_access:
                acc = (handle_prefix + s.handle) if s.handle else ''
                row += f'  {acc:<20}'
            lines.append(row)
            if show_free:
                for p in s.varying:
                    lines.append(f'    + {p}  {_prior_range(p)}')
            if show_fixed:
                for p in s.fixed:
                    lines.append(f'    . {p}  (fixed)')

    multi = getattr(model, 'psls', None) is not None
    for k, (label, ntoa, sigs) in enumerate(collections):
        toa = f'{ntoa} TOAs' if ntoa is not None else ''
        prefix = f'psls[{k}].' if multi else ''
        _emit(sigs, f'{title} - {label}   ({toa})', handle_prefix=prefix)
        lines.append('')

    if commons:
        _emit(commons, 'common / correlated signals')
        lines.append('')

    lines.append('=' * W)
    lines.append(f'pulsars: {len(collections)}   '
                 f'free params: {len(tot["varying"])}   '
                 f'fixed: {len(tot["fixed"])}   '
                 f'common: {len(tot["common"])}')
    lines.append(f'total basis dimension: {tot["basis"]}')
    lines.append('coefficients: how each block appears to the coefficient '
                 'frontend (clogL);')
    lines.append('              the marginal frontend (logL) integrates every '
                 'GP block analytically.')

    text = '\n'.join(lines)
    if to_stdout:
        print(text)
        return None
    return text


def summary_frame(model):
    """One row per signal as a pandas ``DataFrame`` for programmatic use."""
    import pandas as pd

    collections, commons = _collect(model)
    multi = getattr(model, 'psls', None) is not None
    rows = []
    for k, (label, ntoa, sigs) in enumerate(collections):
        prefix = f'psls[{k}].' if multi else ''
        for s in sigs:
            rows.append(dict(collection=label, signal=s.name, kind=s.kind,
                             scope=s.scope, basis=s.basis_shape,
                             n_free=len(s.varying), n_fixed=len(s.fixed),
                             coefficients=s.coefficients,
                             access=(prefix + s.handle) if s.handle else '',
                             free_params=', '.join(s.varying),
                             fixed_params=', '.join(s.fixed)))
    for s in commons:
        rows.append(dict(collection='(shared)', signal=s.name, kind=s.kind,
                         scope=s.scope, basis=s.basis_shape,
                         n_free=len(s.varying), n_fixed=len(s.fixed),
                         coefficients=s.coefficients,
                         access=s.handle or '',
                         free_params=', '.join(s.varying), fixed_params=''))
    return pd.DataFrame(rows, columns=['collection', 'signal', 'kind', 'scope',
                                       'basis', 'n_free', 'n_fixed',
                                       'coefficients', 'access',
                                       'free_params', 'fixed_params'])


def summary_html(model):
    """A notebook-friendly HTML rendering (used by ``_repr_html_``)."""
    collections, commons = _collect(model)
    tot = _totals(collections, commons)

    def _shape(s):
        if not s.basis_shape:
            return '&mdash;'
        if len(s.basis_shape) == 2 and s.scope in ('global', 'common'):
            return f'{s.basis_shape[0]} psr &times; {s.basis_shape[1]}'
        return ' &times; '.join(str(d) for d in s.basis_shape)

    def _rows(sigs):
        out = []
        for s in sigs:
            params = ', '.join(s.varying) or '&mdash;'
            out.append(
                f'<tr><td><code>{s.name}</code></td><td>{s.kind}</td>'
                f'<td style="text-align:right">{_shape(s)}</td>'
                f'<td style="text-align:right">{len(s.varying)}</td>'
                f'<td>{s.coefficients}</td>'
                f'<td style="font-size:90%;color:#555">{params}</td></tr>')
        return ''.join(out)

    head = ('<tr style="text-align:left"><th>signal</th><th>kind</th>'
            '<th style="text-align:right">basis</th>'
            '<th style="text-align:right">free</th><th>coefficients</th>'
            '<th>free parameters</th></tr>')
    blocks = [f'<b>discovery {type(model).__name__}</b> '
              f'<span style="color:#777">({_backend_label()})</span>']
    for label, ntoa, sigs in collections:
        toa = f'{ntoa} TOAs' if ntoa is not None else ''
        blocks.append(f'<div style="margin-top:6px"><b>{label}</b> '
                      f'<span style="color:#777">{toa}</span>'
                      f'<table>{head}{_rows(sigs)}</table></div>')
    if commons:
        blocks.append(f'<div style="margin-top:6px"><b>common / correlated</b>'
                      f'<table>{head}{_rows(commons)}</table></div>')
    blocks.append(
        f'<div style="margin-top:6px;color:#444">pulsars: {len(collections)} &nbsp; '
        f'free: {len(tot["varying"])} &nbsp; fixed: {len(tot["fixed"])} &nbsp; '
        f'common: {len(tot["common"])} &nbsp; basis dim: {tot["basis"]}</div>')
    blocks.append(
        '<div style="margin-top:2px;color:#777;font-size:90%">'
        '<i>coefficients</i>: how each block appears to the coefficient frontend '
        '(<code>clogL</code>); the marginal frontend (<code>logL</code>) '
        'integrates every GP block analytically.</div>')
    return ''.join(blocks)


# --------------------------------------------------------------------------
# kernel tree  (model.tree())
# --------------------------------------------------------------------------

def _render_tree(root_label, children):
    """Render a nested ``(label, children)`` structure with box-drawing rules."""
    lines = [root_label]

    def rec(nodes, prefix):
        for i, (label, subs) in enumerate(nodes):
            last = i == len(nodes) - 1
            lines.append(prefix + ('└─ ' if last else '├─ ') + label)
            if subs:
                rec(subs, prefix + ('   ' if last else '│  '))

    rec(children, '')
    return '\n'.join(lines)


def _shape_str(s):
    if not s.basis_shape:
        return ''
    if len(s.basis_shape) == 2 and s.scope in ('global', 'common'):
        return f'{s.basis_shape[0]}psr×{s.basis_shape[1]}'
    return '×'.join(str(d) for d in s.basis_shape)


def _sig_label(s, handle_prefix=''):
    shape = _shape_str(s)
    handle = (handle_prefix + s.handle) if s.handle else ''
    cols = f'{s.name:<14} {s.kind:<22} {shape:<10}'
    return cols + (f'  {handle}' if handle else '')


def _slice_children(s):
    """Per-pulsar slice leaves for a stacked common/global GP."""
    if not s.slices:
        return []
    return [(f'{psr:<14} → [{sl.start}:{sl.stop}]', [])
            for psr, sl in s.slices.items()]


def _composition_tree(model):
    """One node per signal — how the signals stack into the covariance C."""
    title = type(model).__name__
    collections, commons = _collect(model)
    multi = getattr(model, 'psls', None) is not None

    if not multi:
        label, ntoa, sigs = collections[0]
        toa = f'{ntoa} TOAs' if ntoa is not None else ''
        root = f'{title}  {label}  ({toa})    C = N + Σ FΦFᵀ'
        children = [(_sig_label(s), _slice_children(s)) for s in sigs]
        return _render_tree(root, children)

    nfree = len(_totals(collections, commons)['varying'])
    root = f'{title}  ({len(collections)} pulsars, {nfree} free params)'
    children = []
    for k, (label, ntoa, sigs) in enumerate(collections):
        toa = f'{ntoa} TOAs' if ntoa is not None else ''
        psr_kids = [(_sig_label(s, f'psls[{k}].'), []) for s in sigs]
        children.append((f'{label}  ({toa})  [psls[{k}]]', psr_kids))
    for s in commons:
        children.append((_sig_label(s), _slice_children(s)))
    return _render_tree(root, children)


def _categorize(psl):
    """Split a pulsar's components into (noise kernel, constant GPs, variable
    GPs) exactly as PulsarLikelihood.__init__ does."""
    noise, cgps, vgps = None, [], []
    for obj in getattr(psl, 'signals', []):
        if getattr(obj, 'measurement', None) is not None:
            noise = obj
        elif type(obj).__name__ == 'ConstantGP':
            cgps.append(obj)
        elif type(obj).__name__ == 'VariableGP':
            vgps.append(obj)
    return noise, cgps, vgps


def _layer_node(group, accessor):
    """A single Woodbury layer (one signal, or several fused by concat=True)."""
    infos = [describe_component(g) for g in group]
    if len(group) == 1:
        s = infos[0]
        shape = _shape_str(s)
        return (f'+ {s.name:<14} {s.kind:<22} {shape:<10}  →  {accessor}', [])
    names = ', '.join(s.name for s in infos)
    label = f'+ [{names}] fused  →  {accessor}.F'
    kids, off = [], 0
    for s in infos:
        w = s.basis_shape[1] if s.basis_shape and len(s.basis_shape) == 2 else 0
        kids.append((f'{s.name:<14} {accessor}.F[:, {off}:{off + w}]', []))
        off += w
    return (label, kids)


def _pulsar_literal(psl, prefix=''):
    """Woodbury layering for one pulsar, outermost (model.N) to innermost."""
    noise, cgps, vgps = _categorize(psl)
    concat = getattr(psl, 'concat', True)

    def groups(gps):
        if not gps:
            return []
        return [list(gps)] if concat else [[g] for g in gps]

    # build order wraps constant GPs around the noise, then variable GPs around
    # that; each chained wrap becomes the new outer layer -> reverse for display
    layers = list(reversed(groups(vgps))) + list(reversed(groups(cgps)))

    nodes = []
    for p, group in enumerate(layers):
        nodes.append(_layer_node(group, prefix + 'N' + '.N' * p))
    nacc = prefix + 'N' + '.N' * len(layers)
    ninfo = describe_component(noise) if noise is not None else None
    nodes.append(((ninfo.kind if ninfo else 'noise') + f'  →  {nacc}', []))
    return nodes


def _literal_tree(model):
    """Mirror the live nested Woodbury kernel (model.N.N…)."""
    title = type(model).__name__
    psls = getattr(model, 'psls', None)

    if psls is None:
        root = f'{title}  {getattr(model, "name", "")}  (Woodbury layering, outer→inner)'
        return _render_tree(root, _pulsar_literal(model))

    root = f'{title}  ({len(psls)} pulsars · Woodbury layering per pulsar)'
    children = []
    for k, psl in enumerate(psls):
        children.append((f'{getattr(psl, "name", "psr")}  [psls[{k}]]',
                         _pulsar_literal(psl, prefix=f'psls[{k}].')))
    # commongp / globalgp act at the array level, not inside any psl.N chain
    _, commons = _collect(model)
    for s in commons:
        children.append((_sig_label(s), _slice_children(s)))
    return _render_tree(root, children)


def tree(model, literal=False, to_stdout=False):
    """Visualize the model as a tree.

    With ``literal=False`` (default) this is the *composition* tree — one node
    per signal, showing how the signals stack into the covariance, with the
    live ``signals[i]`` / ``commongp`` / ``globalgp`` handle on each. With
    ``literal=True`` it mirrors the actual nested Woodbury kernel
    (``model.N.N…``), where constant GPs fused by ``concat=True`` appear as a
    single layer with their column slices.
    """
    text = _literal_tree(model) if literal else _composition_tree(model)
    if to_stdout:
        print(text)
        return None
    return text


# --------------------------------------------------------------------------
# model reprs  (tree-based)
# --------------------------------------------------------------------------

_MAX_REPR_PULSARS = 3


def short_repr(model):
    """One-line description of a model (used inside the tree header)."""
    collections, commons = _collect(model)
    tot = _totals(collections, commons)
    if getattr(model, 'psls', None) is None:
        who = collections[0][0] if collections else '?'
        return (f'<{type(model).__name__} {who}: '
                f'{sum(len(s) for _, _, s in collections)} signals, '
                f'{len(tot["varying"])} free params>')
    return (f'<{type(model).__name__}: {len(collections)} pulsars, '
            f'{len(tot["varying"])} free params, {len(commons)} shared signals>')


def repr_tree(model):
    """Composition tree for ``__repr__``; many-pulsar arrays are truncated so an
    accidental echo in the terminal stays bounded."""
    psls = getattr(model, 'psls', None)
    if psls is None or len(psls) <= _MAX_REPR_PULSARS:
        return _composition_tree(model)

    # truncate: keep the first few pulsar subtrees + the shared branches
    title = type(model).__name__
    collections, commons = _collect(model)
    nfree = len(_totals(collections, commons)['varying'])
    root = f'{title}  ({len(collections)} pulsars, {nfree} free params)'
    children = []
    for k, (label, ntoa, sigs) in enumerate(collections[:_MAX_REPR_PULSARS]):
        toa = f'{ntoa} TOAs' if ntoa is not None else ''
        psr_kids = [(_sig_label(s, f'psls[{k}].'), []) for s in sigs]
        children.append((f'{label}  ({toa})  [psls[{k}]]', psr_kids))
    children.append((f'… ({len(collections) - _MAX_REPR_PULSARS} more pulsars — '
                     f'model.tree() for all)', []))
    for s in commons:
        children.append((_sig_label(s), _slice_children(s)))
    return _render_tree(root, children)


class SummaryMixin:
    """Adds ``summary`` / ``summary_frame`` / ``tree`` / notebook + text reprs to
    a likelihood class. The heavy lifting lives in the module-level functions so
    both backend class families (matrix and metamath) can share one mixin."""

    def summary(self, include_params=None, *, show_free=True, show_fixed=True,
                show_access=False, to_stdout=False):
        return summary(self, include_params=include_params, show_free=show_free,
                       show_fixed=show_fixed, show_access=show_access,
                       to_stdout=to_stdout)

    def summary_frame(self):
        return summary_frame(self)

    def tree(self, literal=False, to_stdout=False):
        return tree(self, literal=literal, to_stdout=to_stdout)

    def _repr_html_(self):
        # notebook display: the composition tree in full (monospaced so the
        # box-drawing aligns). The signal table stays available via
        # .summary() / .summary_frame().
        import html
        return (f'<pre style="line-height:1.35;font-size:90%">'
                f'{html.escape(_composition_tree(self))}</pre>')

    def __repr__(self):
        return repr_tree(self)
