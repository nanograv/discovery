"""`discovery.samplers.numpyro` — kwarg routing and checkpoint-runner hardening.

The old `makesampler_nuts` iterated `kwargs.items()` and tested each `(key,
value)` tuple for membership in an argname list, so the test was always false
and every kwarg was silently discarded. These tests pin the routing.
"""
import pickle

import pandas as pd
import pytest

import jax
jax.config.update("jax_enable_x64", True)

import numpyro  # noqa: E402
from numpyro import distributions as dist  # noqa: E402
from numpyro import infer  # noqa: E402

from discovery.samplers import numpyro as ds_numpyro  # noqa: E402


@pytest.fixture
def toy_model():
    """A minimal NumPyro model carrying the `to_df` attribute that
    `makesampler_nuts` / `run_nuts_with_checkpoints` rely on."""
    def numpyro_model():
        numpyro.sample("x", dist.Normal(0.0, 1.0))

    numpyro_model.to_df = lambda chain: pd.DataFrame({"x": chain["x"]})
    return numpyro_model


class TestMakesamplerNuts:
    def test_defaults_reach_nuts_and_mcmc(self, toy_model):
        sampler = ds_numpyro.makesampler_nuts(toy_model)

        assert sampler.sampler._max_tree_depth == 8
        assert sampler.sampler._dense_mass is False
        assert sampler.sampler._forward_mode_differentiation is False
        assert sampler.sampler._target_accept_prob == 0.8

        assert sampler.num_warmup == 512
        assert sampler.num_samples == 1024
        assert sampler.num_chains == 1
        assert sampler.chain_method == "vectorized"

    def test_nuts_override_is_honored(self, toy_model):
        """The regression: `max_tree_depth=6` used to be dropped silently."""
        sampler = ds_numpyro.makesampler_nuts(toy_model, max_tree_depth=6)

        assert sampler.sampler._max_tree_depth == 6

    def test_mcmc_override_is_honored(self, toy_model):
        sampler = ds_numpyro.makesampler_nuts(toy_model, num_warmup=7,
                                              progress_bar=False)

        assert sampler.num_warmup == 7
        assert sampler.progress_bar is False

    def test_unknown_kwarg_raises_typeerror_naming_it(self, toy_model):
        with pytest.raises(TypeError, match="max_tree_dpeth"):
            ds_numpyro.makesampler_nuts(toy_model, max_tree_dpeth=6)

    def test_potential_fn_raises_because_a_model_is_supplied(self, toy_model):
        """`potential_fn` is in the NUTS signature but is mutually exclusive
        with the positional model this factory always passes."""
        with pytest.raises(TypeError, match="potential_fn"):
            ds_numpyro.makesampler_nuts(toy_model, potential_fn=lambda z: 0.0)

    def test_to_df_delegates_to_the_model(self, toy_model):
        sampler = ds_numpyro.makesampler_nuts(toy_model, num_warmup=2,
                                              num_samples=3, progress_bar=False)
        sampler.run(jax.random.PRNGKey(0))

        df = sampler.to_df()

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["x"]
        assert len(df) == 3


class TestEnsureSamplerToDf:
    def test_noop_when_already_present(self, toy_model):
        sampler = ds_numpyro.makesampler_nuts(toy_model)
        before = sampler.to_df

        ds_numpyro._ensure_sampler_to_df(sampler)

        assert sampler.to_df is before

    def test_attaches_from_sampler_sampler_model(self, toy_model):
        """A raw MCMC around a to_df-carrying model gets the attachment back."""
        sampler = infer.MCMC(infer.NUTS(toy_model), num_warmup=2, num_samples=3,
                             progress_bar=False)
        assert not hasattr(sampler, "to_df")

        ds_numpyro._ensure_sampler_to_df(sampler)
        sampler.run(jax.random.PRNGKey(0))

        assert list(sampler.to_df().columns) == ["x"]

    def test_raises_attributeerror_without_a_to_df_model(self):
        def bare_model():
            numpyro.sample("x", dist.Normal(0.0, 1.0))

        sampler = infer.MCMC(infer.NUTS(bare_model), num_warmup=2,
                             num_samples=3, progress_bar=False)

        with pytest.raises(AttributeError, match="has no to_df"):
            ds_numpyro._ensure_sampler_to_df(sampler)


class TestRunNutsWithCheckpoints:
    def test_creates_missing_nested_outdir_and_appends_across_rounds(
            self, toy_model, tmp_path):
        sampler = ds_numpyro.makesampler_nuts(toy_model, num_warmup=4,
                                              num_samples=4, progress_bar=False)
        outdir = tmp_path / "deeply" / "nested" / "chains"
        assert not outdir.exists()

        ds_numpyro.run_nuts_with_checkpoints(sampler, 2, jax.random.PRNGKey(0),
                                             outdir=outdir)

        assert outdir.is_dir()
        # Two checkpoint rounds of 2 samples each, appended into one frame.
        df = pd.read_feather(outdir / "numpyro-samples.feather")
        assert len(df) == 4

        # The checkpoint holds sampler state, not model objects.
        with (outdir / "numpyro-checkpoint.pickle").open("rb") as f:
            pickle.load(f)

    def test_accepts_a_path_outdir_that_does_not_exist(self, toy_model, tmp_path):
        """The old body only created `outdir` when it was *not* already a Path,
        so passing a Path to a missing directory failed."""
        sampler = ds_numpyro.makesampler_nuts(toy_model, num_warmup=2,
                                              num_samples=2, progress_bar=False)
        outdir = tmp_path / "as_path"

        ds_numpyro.run_nuts_with_checkpoints(sampler, 2, jax.random.PRNGKey(0),
                                             outdir=outdir)

        assert (outdir / "numpyro-samples.feather").is_file()


# --- tests brought in from main (recording / monkeypatch style) ---

import pytest

import discovery.samplers.numpyro as ds_numpyro


class RecordingNUTS:
    def __init__(
        self,
        model=None,
        *,
        max_tree_depth=10,
        dense_mass=False,
        forward_mode_differentiation=False,
        target_accept_prob=0.8,
        init_strategy=None,
    ):
        self.model = model
        self.options = {
            "max_tree_depth": max_tree_depth,
            "dense_mass": dense_mass,
            "forward_mode_differentiation": forward_mode_differentiation,
            "target_accept_prob": target_accept_prob,
            "init_strategy": init_strategy,
        }


class RecordingMCMC:
    def __init__(
        self,
        sampler,
        *,
        num_warmup,
        num_samples,
        num_chains=1,
        thinning=1,
        chain_method="parallel",
        progress_bar=True,
    ):
        self.sampler = sampler
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.thinning = thinning
        self.chain_method = chain_method
        self.progress_bar = progress_bar

    def get_samples(self):
        return {"pars": "samples"}


def dummy_model():
    pass


dummy_model.to_df = lambda samples: ("dataframe", samples)


@pytest.fixture
def recording_infer(monkeypatch):
    monkeypatch.setattr(ds_numpyro.infer, "NUTS", RecordingNUTS)
    monkeypatch.setattr(ds_numpyro.infer, "MCMC", RecordingMCMC)


def test_makesampler_nuts_forwards_overrides_and_preserves_defaults(recording_infer):
    init = object()
    sampler = ds_numpyro.makesampler_nuts(
        dummy_model,
        num_warmup=7,
        num_samples=11,
        num_chains=2,
        dense_mass=True,
        target_accept_prob=0.91,
        init_strategy=init,
        thinning=3,
        chain_method="parallel",
        progress_bar=False,
    )

    assert isinstance(sampler, RecordingMCMC)
    assert isinstance(sampler.sampler, RecordingNUTS)
    assert sampler.sampler.model is dummy_model
    assert sampler.sampler.options == {
        "max_tree_depth": 8,
        "dense_mass": True,
        "forward_mode_differentiation": False,
        "target_accept_prob": 0.91,
        "init_strategy": init,
    }
    assert sampler.num_warmup == 7
    assert sampler.num_samples == 11
    assert sampler.num_chains == 2
    assert sampler.thinning == 3
    assert sampler.chain_method == "parallel"
    assert sampler.progress_bar is False
    assert sampler.to_df() == ("dataframe", {"pars": "samples"})


def test_makesampler_nuts_defaults_unchanged(recording_infer):
    sampler = ds_numpyro.makesampler_nuts(dummy_model)

    assert sampler.sampler.options == {
        "max_tree_depth": 8,
        "dense_mass": False,
        "forward_mode_differentiation": False,
        "target_accept_prob": 0.8,
        "init_strategy": None,
    }
    assert sampler.num_warmup == 512
    assert sampler.num_samples == 1024
    assert sampler.num_chains == 1
    assert sampler.thinning == 1
    assert sampler.chain_method == "vectorized"
    assert sampler.progress_bar is True


def test_makesampler_nuts_rejects_unknown_keyword(recording_infer):
    with pytest.raises(TypeError, match="target_accept_probability"):
        ds_numpyro.makesampler_nuts(dummy_model, target_accept_probability=0.91)


def test_ensure_sampler_to_df_noop_when_already_present():
    sampler = RecordingMCMC(RecordingNUTS(dummy_model), num_warmup=1, num_samples=1)
    sampler.to_df = lambda: "already"
    ds_numpyro._ensure_sampler_to_df(sampler)
    assert sampler.to_df() == "already"


def test_ensure_sampler_to_df_recovers_from_kernel_model():
    sampler = RecordingMCMC(RecordingNUTS(dummy_model), num_warmup=1, num_samples=1)
    assert not hasattr(sampler, "to_df")
    ds_numpyro._ensure_sampler_to_df(sampler)
    assert sampler.to_df() == ("dataframe", {"pars": "samples"})


def test_ensure_sampler_to_df_raises_without_model_to_df():
    class BareKernel:
        pass

    sampler = RecordingMCMC(BareKernel(), num_warmup=1, num_samples=1)
    with pytest.raises(AttributeError, match="makesampler_nuts"):
        ds_numpyro._ensure_sampler_to_df(sampler)


def test_run_nuts_with_checkpoints_mkdirs_path_and_recovers_to_df(tmp_path, monkeypatch):
    import pandas as pd

    def model_with_df():
        pass

    model_with_df.to_df = lambda samples: pd.DataFrame(
        {"x": [samples["chunk"]]}
    )

    class FakeSampler:
        def __init__(self):
            self.num_samples = 4
            self.sampler = RecordingNUTS(model_with_df)
            self.last_state = "state"
            self.post_warmup_state = None
            self.runs = 0

        def _set_collection_params(self):
            pass

        def run(self, rng_key):
            self.runs += 1

        def get_samples(self):
            return {"chunk": self.runs}

    saved = {}

    def fake_save_chain(df, path):
        saved["df"] = df.copy()
        saved["path"] = path

    monkeypatch.setattr(ds_numpyro, "save_chain", fake_save_chain)
    monkeypatch.setattr(
        ds_numpyro.jax.random, "split", lambda key: (key, key)
    )

    outdir = tmp_path / "nested" / "chains"
    sampler = FakeSampler()
    assert not hasattr(sampler, "to_df")

    df = ds_numpyro.run_nuts_with_checkpoints(
        sampler,
        num_samples_per_checkpoint=2,
        rng_key=0,
        outdir=outdir,
        resume=False,
    )

    assert outdir.is_dir()
    assert callable(sampler.to_df)
    assert sampler.runs == 2
    assert saved["path"] == outdir / "numpyro-samples.feather"
    assert list(df["x"]) == [1, 2]
    assert (outdir / "numpyro-checkpoint.pickle").is_file()
