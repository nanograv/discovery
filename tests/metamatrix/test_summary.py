"""Model-summary snapshots (discovery.summary).

A built likelihood can describe itself via ``model.summary()`` /
``summary_frame()`` / ``_repr_html_``. These tests exercise every recipe under
both kernel backends and assert the one invariant that makes the summary
trustworthy: the free parameters it reports are a *superset* of
``logL.params`` -- it never hides a parameter the likelihood actually varies.
The only allowed extras are deterministic-signal and non-zero-mean parameters,
which some likelihood paths marginalize out of ``logL.params`` but which are
still genuine model structure worth showing.
"""

import pandas as pd
import pytest

import jax

import discovery as ds
import discovery.recipes as R
from discovery import summary as S


@pytest.fixture(params=["matrix", "metamath"])
def backend(request):
    """Run each test under both kernel backends, restoring matrix afterwards."""
    ds.config(kernels=request.param)
    yield request.param
    ds.config(kernels="matrix")


def _extra_allowed(model):
    """Params the summary may report beyond logL.params: extsignal coefficients
    and non-zero prior-mean amplitudes (handled outside the main param vector)."""
    extra = set()
    for ext in getattr(model, "extsignals", None) or []:
        extra.update(getattr(ext, "params", []))
    cg = getattr(model, "commongp", None)
    for g in (cg if isinstance(cg, list) else [cg]) if cg is not None else []:
        for attr in ("means", "mean"):
            fn = getattr(g, attr, None)
            if fn is not None:
                extra.update(getattr(fn, "params", []))
    return extra


def _summary_params(model):
    cols, com = S._collect(model)
    return set(S._totals(cols, com)["varying"])


SINGLE = [pytest.param(f, id=f.__name__) for f in R.SINGLE_PULSAR]
MULTI = [pytest.param(f, id=f.__name__) for f in (R.GLOBAL + R.ARRAY)]


@pytest.mark.parametrize("recipe", SINGLE)
def test_single_pulsar_summary(recipe, psr, backend):
    model = recipe(psr)

    # the reliability invariant: never hide a varied parameter
    assert set(model.logL.params) <= _summary_params(model)
    assert _summary_params(model) - set(model.logL.params) <= _extra_allowed(model)

    text = model.summary()
    assert isinstance(text, str) and psr.name in text
    assert "free params" in text

    frame = model.summary_frame()
    assert isinstance(frame, pd.DataFrame)
    assert {"signal", "kind", "basis", "n_free", "access"} <= set(frame.columns)
    assert len(frame) >= 1

    assert "<pre" in model._repr_html_()
    assert isinstance(repr(model), str) and type(model).__name__ in repr(model)

    # kernel tree (both renderings) names the pulsar and carries a handle
    comp = model.tree()
    assert isinstance(comp, str) and psr.name in comp and "signals[" in comp
    assert isinstance(model.tree(literal=True), str)

    # independent free/fixed toggles
    assert "(fixed)" in model.summary(show_fixed=True, show_free=False) or \
        frame["n_fixed"].sum() == 0
    assert "(fixed)" not in model.summary(show_fixed=False)


@pytest.mark.parametrize("recipe", MULTI)
def test_multi_pulsar_summary(recipe, psrs, backend):
    model = recipe(psrs)

    assert set(model.logL.params) <= _summary_params(model)
    assert _summary_params(model) - set(model.logL.params) <= _extra_allowed(model)

    text = model.summary()
    assert isinstance(text, str)
    for p in psrs:
        assert p.name in text

    frame = model.summary_frame()
    assert isinstance(frame, pd.DataFrame)
    # one collection per pulsar should appear
    assert set(p.name for p in psrs) <= set(frame["collection"])

    assert "<pre" in model._repr_html_()

    # kernel tree: every pulsar named, per-pulsar handles present
    comp = model.tree()
    for p in psrs:
        assert p.name in comp
    assert "psls[0].signals[" in comp
    assert isinstance(model.tree(literal=True), str)


def test_fixed_white_noise_is_reported(psr):
    """White noise pinned from the noise dictionary is invisible to logL.params
    but must still show up as fixed parameters in the summary."""
    ds.config(kernels="matrix")
    model = R.full_rn(psr)
    frame = model.summary_frame()
    meas = frame[frame["signal"] == "measurement"].iloc[0]
    assert meas["n_free"] == 0
    assert meas["n_fixed"] > 0          # efac / equad baked in from noisedict
    assert f"{psr.name}_" in meas["fixed_params"]


@pytest.fixture
def _matrix_backend():
    ds.config(kernels="matrix")
    yield
    ds.config(kernels="matrix")


def test_literal_tree_fuses_constant_gps(psr, _matrix_backend):
    """concat=True fuses ECORR+timing into one Woodbury layer; the literal tree
    must show their column slices summing to the live model.N.N.F width."""
    model = R.full_rn(psr)
    lit = model.tree(literal=True)
    assert "fused" in lit
    # ECORR (360) then timing (166) -> 526, matching the live fused basis
    assert model.N.N.F.shape[1] == 526
    assert "[:, 0:360]" in lit and "[:, 360:526]" in lit

    # concat=False chains them into separate layers (no fusion)
    assert "fused" not in R.full_rn_concat_false(psr).tree(literal=True)


def test_signal_object_reprs(psr, psrs, _matrix_backend):
    """signals[i] / commongp / globalgp print an informative repr, not <object>."""
    model = R.full_rn(psr)
    rn = model.signals[4]
    assert "rednoise" in repr(rn) and "VariableGP" in repr(rn)
    assert "fixed prior" in repr(model.signals[3])     # timing model (ConstantGP)
    assert "white noise" in repr(model.signals[1])     # measurement kernel

    g = R.intrinsic_rn_plus_global_hd(psrs)
    assert "GlobalVariableGP" in repr(g.globalgp) and "hd_orf" in repr(g.globalgp)


# --------------------------------------------------------------------------
# the `coefficients` column: treatment is a property of the FRONTEND,
# derived from the assembled kernel's .index -- never from the GP's type.
# --------------------------------------------------------------------------

def _row(frame, signal, collection=None):
    sel = frame[frame.signal == signal]
    if collection is not None:
        sel = sel[sel.collection == collection]
    assert len(sel) == 1, f"expected one '{signal}' row, got {len(sel)}"
    return sel.iloc[0]


def test_variable_timing_reports_sampled_timing_and_kernel_white_noise(psr, backend):
    """`makegp_timing(variable=True)` exposes sampled coefficients to clogL; the
    white-noise kernel is not a coefficient block at all."""
    frame = S.summary_frame(R.variable_timing(psr))

    assert _row(frame, "timingmodel").coefficients.startswith("sampled (")
    assert _row(frame, "measurement").coefficients == "kernel"
    # rednoise is variable too and concat=True fuses both into one index
    assert _row(frame, "rednoise").coefficients.startswith("sampled (")


def test_shadowed_gp_is_reported_marginalized_not_sampled(psr, backend):
    """The mislabeling regression: under `concat=False` only the LAST variable
    GP keeps sampled coefficients. Reporting from the GP's type would call both
    'sampled'; reporting from the assembled index tells the truth."""
    model = ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_fourier(psr, ds.powerlaw, components=10, name="rednoise"),
        ds.makegp_fourier(psr, ds.powerlaw, components=5, name="crn"),
    ], concat=False, marginalize_all_but_last=True)

    frame = S.summary_frame(model)

    assert _row(frame, "crn").coefficients.startswith("sampled (")
    assert _row(frame, "rednoise").coefficients == "marginalized"


def test_inner_gp_is_marginalized_when_an_outer_commongp_assembles(psrs, backend):
    """An ArrayLikelihood with an outer coefficient assembly marginalizes the
    per-pulsar GPs inside each psl.N; only the outer commongp is sampled."""
    T = ds.getspan(psrs)
    model = ds.ArrayLikelihood(
        [ds.PulsarLikelihood([
            p.residuals,
            ds.makenoise_measurement(p, p.noisedict),
            ds.makegp_timing(p, svd=True),
            # a VARIABLE GP inside the per-pulsar kernel
            ds.makegp_fourier(p, ds.powerlaw, components=10, name="innerrn"),
        ]) for p in psrs],
        commongp=ds.makecommongp_fourier(psrs, ds.powerlaw, components=30, T=T,
                                         name="rednoise"))

    frame = S.summary_frame(model)

    assert _row(frame, "rednoise", collection="(shared)").coefficients.startswith("sampled (")
    for p in psrs:
        assert _row(frame, "innerrn", collection=p.name).coefficients == "marginalized"


def test_no_outer_assembly_reports_each_pulsars_own_index(psrs, backend):
    """Without commongp/globalgp, ArrayLikelihood.clogL is the sum of psl.clogL,
    so each pulsar's own variable GP IS sampled."""
    model = ds.ArrayLikelihood([
        ds.PulsarLikelihood([
            p.residuals,
            ds.makenoise_measurement(p, p.noisedict),
            ds.makegp_timing(p, svd=True),
            ds.makegp_fourier(p, ds.powerlaw, components=10, name="innerrn"),
        ]) for p in psrs])

    frame = S.summary_frame(model)

    for p in psrs:
        assert _row(frame, "innerrn", collection=p.name).coefficients.startswith("sampled (")


def test_projected_timing_is_reported_projected(psr, backend):
    """A project=True GP is projected out of the kernel entirely, so
    it is neither sampled nor marginalized."""
    model = ds.PulsarLikelihood([
        psr.residuals,
        ds.makenoise_measurement(psr, psr.noisedict),
        ds.makegp_ecorr(psr, psr.noisedict),
        ds.makegp_timing(psr, svd=True, project=True),
    ])

    assert _row(S.summary_frame(model), "timingmodel").coefficients == "projected"


def test_extsignal_row_is_deterministic(psrs, backend):
    frame = S.summary_frame(R.extsignal_cw(psrs))
    cw = frame[frame.scope == "external"]

    assert len(cw) == 1
    assert cw.iloc[0].coefficients == "deterministic"


def test_column_is_rendered_in_text_and_html(psr, backend):
    model = R.variable_timing(psr)

    text = model.summary(show_free=False, show_fixed=False)
    assert "coefficients" in text
    assert "sampled (" in text and "kernel" in text
    assert "the marginal frontend (logL) integrates every GP block analytically." in text

    html = S.summary_html(model)
    assert "<th>coefficients</th>" in html
