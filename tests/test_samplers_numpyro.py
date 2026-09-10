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
