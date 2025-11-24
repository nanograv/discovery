import numpy as np
import pytest
from discovery import signals, matrix
from pathlib import Path

class MockPulsar:
    """Mock pulsar object for testing"""
    def __init__(self, name, ntoas=100, nbackends=2):
        self.name = name
        self.toas = np.sort(np.random.uniform(0, 1000, ntoas))
        self.toaerrs = np.random.uniform(0.1, 1.0, ntoas)

        # Create backend flags
        backends = [f'backend{i}' for i in range(nbackends)]
        self.backend_flags = np.random.choice(backends, size=ntoas)

        # Add position (needed for some GP functions)
        self.pos = np.array([1.0, 0.0, 0.0])  # arbitrary unit vector

    @property
    def residuals(self):
        return np.random.randn(len(self.toas))

@pytest.mark.unit
def test_makenoise_measurement_fixed_params():
    """Test makenoise_measurement with fixed noise parameters"""
    psr = MockPulsar('J0437-4715', ntoas=100, nbackends=2)
    backends = sorted(set(psr.backend_flags))

    # Create fixed noise dictionary
    noisedict = {}
    for backend in backends:
        noisedict[f'{psr.name}_{backend}_efac'] = 1.2
        noisedict[f'{psr.name}_{backend}_log10_t2equad'] = -2.5

    # Get noise matrix
    noise = signals.makenoise_measurement(psr, noisedict=noisedict, ecorr=False)

    # Should return NoiseMatrix1D_novar since all params are fixed
    assert isinstance(noise, matrix.NoiseMatrix1D_novar)
    assert noise.N.shape == (psr.toas.shape[0],)
    assert np.all(noise.N > 0)  # All noise values should be positive

@pytest.mark.unit
def test_makenoise_measurement_variable_params():
    """Test makenoise_measurement with variable noise parameters"""
    psr = MockPulsar('J0437-4715', ntoas=100, nbackends=2)
    backends = sorted(set(psr.backend_flags))

    # Don't provide noise dict -> parameters are variable
    noise = signals.makenoise_measurement(psr, noisedict={}, ecorr=False)

    # Should return NoiseMatrix1D_var since params need to be sampled
    assert isinstance(noise, matrix.NoiseMatrix1D_var)
    assert hasattr(noise, 'getN')
    assert hasattr(noise, 'params')

    # Check expected parameters
    expected_params = []
    for backend in backends:
        expected_params.append(f'{psr.name}_{backend}_efac')
        expected_params.append(f'{psr.name}_{backend}_log10_t2equad')
    assert sorted(noise.params) == sorted(expected_params)

@pytest.mark.unit
def test_makenoise_measurement_with_ecorr():
    """Test makenoise_measurement with ECORR enabled"""
    psr = MockPulsar('J0437-4715', ntoas=100, nbackends=2)
    backends = sorted(set(psr.backend_flags))

    noisedict = {}
    for backend in backends:
        noisedict[f'{psr.name}_{backend}_efac'] = 1.0
        noisedict[f'{psr.name}_{backend}_log10_t2equad'] = -2.5
        noisedict[f'{psr.name}_{backend}_log10_ecorr'] = -2.5

    noise = signals.makenoise_measurement(psr, noisedict=noisedict, ecorr=True)

    # Should return NoiseMatrixSM_novar (Sherman-Morrison form)
    assert isinstance(noise, matrix.NoiseMatrixSM_novar)
    assert hasattr(noise, 'N')  # diagonal component
    assert hasattr(noise, 'F')  # low-rank component basis
    assert hasattr(noise, 'P')  # low-rank component covariance

@pytest.mark.unit
def test_makenoise_measurement_tnequad():
    """Test makenoise_measurement with TNEQUAD instead of T2EQUAD"""
    psr = MockPulsar('J0437-4715', ntoas=100, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    noisedict = {
        f'{psr.name}_{backend}_efac': 1.3,
        f'{psr.name}_{backend}_log10_tnequad': -2.5
    }

    noise = signals.makenoise_measurement(psr, noisedict=noisedict, tnequad=True)

    assert isinstance(noise, matrix.NoiseMatrix1D_novar)
    assert noise.N.shape == (psr.toas.shape[0],)

@pytest.mark.unit
def test_makenoise_measurement_scale():
    """Test that scale parameter affects noise correctly"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    # Set efac=1.0 and very small equad so noise is dominated by scaled toaerrs
    noisedict = {
        f'{psr.name}_{backend}_efac': 1.0,
        f'{psr.name}_{backend}_log10_t2equad': -20.0  # Very small equad
    }

    scale1 = 1.0
    scale2 = 2.5

    noise1 = signals.makenoise_measurement(psr, noisedict=noisedict, scale=scale1)
    noise2 = signals.makenoise_measurement(psr, noisedict=noisedict, scale=scale2)

    # Expected: N = efac^2 * (scale * toaerrs)^2 + 10^(2 * (log10(scale) + log10_t2equad))
    # With efac=1 and tiny equad: N ≈ (scale * toaerrs)^2
    # So ratio should be (scale2/scale1)^2
    expected_ratio = (scale2 / scale1) ** 2
    actual_ratio = noise2.N / noise1.N

    # Check ratio is close to expected (with small tolerance for equad contribution)
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-6)

    # Also verify the absolute values match expected formula
    equad2 = 10.0 ** (2.0 * noisedict[f'{psr.name}_{backend}_log10_t2equad'])
    expected_noise1 = (scale1 * psr.toaerrs) ** 2 + equad2
    expected_noise2 = (scale2 * psr.toaerrs) ** 2 + equad2

    np.testing.assert_allclose(noise1.N, expected_noise1, rtol=1e-10)
    np.testing.assert_allclose(noise2.N, expected_noise2, rtol=1e-10)

@pytest.mark.unit
def test_makenoise_measurement_single_backend():
    """Test edge case with single backend (regression test for PR #99)"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)

    # All observations should have the same backend
    assert len(set(psr.backend_flags)) == 1

    backend = list(set(psr.backend_flags))[0]
    noisedict = {
        f'{psr.name}_{backend}_efac': 1.1,
        f'{psr.name}_{backend}_log10_t2equad': -2.5
    }

    # Should work without errors
    noise = signals.makenoise_measurement(psr, noisedict=noisedict)
    assert isinstance(noise, matrix.NoiseMatrix1D_novar)

@pytest.mark.unit
def test_makenoise_measurement_vectorize():
    """Test vectorize parameter affects implementation but gives same results"""
    np.random.seed(1)
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=2)

    noise_vec = signals.makenoise_measurement(psr, noisedict={}, vectorize=True, ecorr=False)
    noise_novec = signals.makenoise_measurement(psr, noisedict={}, vectorize=False, ecorr=False)

    # Both should be NoiseMatrix1D_var
    assert isinstance(noise_vec, matrix.NoiseMatrix1D_var)
    assert isinstance(noise_novec, matrix.NoiseMatrix1D_var)

    # Both should have the same parameters
    assert noise_vec.params == noise_novec.params

    # Test they give same results (create sample params)
    backends = sorted(set(psr.backend_flags))
    params = {}
    for backend in backends:
        params[f'{psr.name}_{backend}_efac'] = 1.2
        params[f'{psr.name}_{backend}_log10_t2equad'] = -2.5

    result_vec = noise_vec.getN(params)
    result_novec = noise_novec.getN(params)
    print(result_vec, result_novec)
    np.testing.assert_allclose(result_vec, result_novec, rtol=1e-10)


@pytest.mark.unit
def test_makenoise_measurement_tnequad_equivalence():
    """Test that tnequad and t2equad are equivalent when efac=1"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    # With efac=1, tnequad and t2equad should give the same results
    log10_equad_value = -2.5

    noisedict_tnequad = {
        f'{psr.name}_{backend}_efac': 1.0,
        f'{psr.name}_{backend}_log10_tnequad': log10_equad_value
    }

    noisedict_t2equad = {
        f'{psr.name}_{backend}_efac': 1.0,
        f'{psr.name}_{backend}_log10_t2equad': log10_equad_value
    }

    noise_tnequad = signals.makenoise_measurement(psr, noisedict=noisedict_tnequad, tnequad=True)
    noise_t2equad = signals.makenoise_measurement(psr, noisedict=noisedict_t2equad, tnequad=False)

    # Both should be NoiseMatrix1D_novar
    assert isinstance(noise_tnequad, matrix.NoiseMatrix1D_novar)
    assert isinstance(noise_t2equad, matrix.NoiseMatrix1D_novar)

    # With efac=1, the noise values should be identical
    np.testing.assert_allclose(noise_tnequad.N, noise_t2equad.N, rtol=1e-10)


@pytest.mark.unit
def test_makenoise_measurement_tnequad_variable():
    """Test tnequad with variable parameters (vectorized and non-vectorized)"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=2)
    backends = sorted(set(psr.backend_flags))

    # Test vectorized version
    noise_vec = signals.makenoise_measurement(psr, noisedict={}, tnequad=True, vectorize=True)
    assert isinstance(noise_vec, matrix.NoiseMatrix1D_var)

    # Check expected parameters
    expected_params = []
    for backend in backends:
        expected_params.append(f'{psr.name}_{backend}_efac')
        expected_params.append(f'{psr.name}_{backend}_log10_tnequad')
    assert sorted(noise_vec.params) == sorted(expected_params)

    # Test non-vectorized version
    noise_novec = signals.makenoise_measurement(psr, noisedict={}, tnequad=True, vectorize=False)
    assert isinstance(noise_novec, matrix.NoiseMatrix1D_var)
    assert sorted(noise_novec.params) == sorted(expected_params)

    # Test they give same results
    params = {}
    for backend in backends:
        params[f'{psr.name}_{backend}_efac'] = 1.2
        params[f'{psr.name}_{backend}_log10_tnequad'] = -2.5

    result_vec = noise_vec.getN(params)
    result_novec = noise_novec.getN(params)

    np.testing.assert_allclose(result_vec, result_novec, rtol=1e-10)


@pytest.mark.unit
def test_makenoise_measurement_outliers():
    """Test outliers parameter scales noise by alpha parameter"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    # Create model with outliers, efac=1, very small equad
    # So noise is dominated by alpha * toaerrs
    noise = signals.makenoise_measurement(psr, noisedict={}, outliers=True)

    # Should be NoiseMatrix1D_var
    assert isinstance(noise, matrix.NoiseMatrix1D_var)

    # Check that alpha_scaling parameter is included
    alpha_param = f'{psr.name}_alpha_scaling({psr.toas.size})'
    assert alpha_param in noise.params

    # Test with different alpha values
    # Note: alpha_scaling multiplies the squared toaerrs: N = efac^2 * alpha_scaling * toaerrs^2 + equad^2
    alpha1 = np.ones(psr.toas.size)
    alpha2 = 2.5 * np.ones(psr.toas.size)

    params_base = {
        f'{psr.name}_{backend}_efac': 1.0,
        f'{psr.name}_{backend}_log10_t2equad': -20.0,  # Very small equad
    }

    params1 = {**params_base, alpha_param: alpha1}
    params2 = {**params_base, alpha_param: alpha2}

    noise1 = noise.getN(params1)
    noise2 = noise.getN(params2)

    # Verify absolute values: N = efac^2 * alpha_scaling * toaerrs^2 + equad^2
    # With efac=1 and very small equad: N ≈ alpha_scaling * toaerrs^2
    equad2 = 10.0 ** (2.0 * params_base[f'{psr.name}_{backend}_log10_t2equad'])
    expected_noise1 = alpha1 * psr.toaerrs ** 2 + equad2
    expected_noise2 = alpha2 * psr.toaerrs ** 2 + equad2

    np.testing.assert_allclose(noise1, expected_noise1, rtol=1e-10)
    np.testing.assert_allclose(noise2, expected_noise2, rtol=1e-10)


@pytest.mark.unit
def test_makenoise_measurement_outliers_error_fixed():
    """Test that outliers=True raises error when noise is fixed"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    # Need to include the alpha parameter in noisedict to make it "fixed"
    alpha_param = f'{psr.name}_alpha_scaling({psr.toas.size})'
    noisedict = {
        f'{psr.name}_{backend}_efac': 1.0,
        f'{psr.name}_{backend}_log10_t2equad': -2.5,
        alpha_param: np.ones(psr.toas.size),
    }

    # Should raise ValueError when outliers=True with fixed noise
    with pytest.raises(ValueError, match="No outlier scaling if white noise is fixed"):
        signals.makenoise_measurement(psr, noisedict=noisedict, outliers=True)

@pytest.mark.unit
def test_makenoise_measurement_simple_consistency():
    """Test that makenoise_measurement_simple gives same results as makenoise_measurement
    when using a single backend with scale=1.0, no tnequad, no ecorr, no outliers"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    # Test with fixed parameters
    noisedict_simple = {
        f'{psr.name}_efac': 1.2,
        f'{psr.name}_log10_t2equad': -2.5
    }
    noisedict_full = {
        f'{psr.name}_{backend}_efac': 1.2,
        f'{psr.name}_{backend}_log10_t2equad': -2.5
    }

    noise_simple = signals.makenoise_measurement_simple(psr, noisedict=noisedict_simple)
    noise_full = signals.makenoise_measurement(psr, noisedict=noisedict_full, scale=1.0)

    assert isinstance(noise_simple, matrix.NoiseMatrix1D_novar)
    assert isinstance(noise_full, matrix.NoiseMatrix1D_novar)
    np.testing.assert_allclose(noise_simple.N, noise_full.N, rtol=1e-10)

    # Test with variable parameters
    noise_simple_var = signals.makenoise_measurement_simple(psr, noisedict={})
    noise_full_var = signals.makenoise_measurement(psr, noisedict={}, scale=1.0)

    assert isinstance(noise_simple_var, matrix.NoiseMatrix1D_var)
    assert isinstance(noise_full_var, matrix.NoiseMatrix1D_var)

    # Check parameters match
    assert set(noise_simple_var.params) == {f'{psr.name}_efac', f'{psr.name}_log10_t2equad'}
    assert set(noise_full_var.params) == {f'{psr.name}_{backend}_efac', f'{psr.name}_{backend}_log10_t2equad'}

    # Test they give same results with corresponding parameter names
    params_simple = {
        f'{psr.name}_efac': 1.2,
        f'{psr.name}_log10_t2equad': -2.5
    }
    params_full = {
        f'{psr.name}_{backend}_efac': 1.2,
        f'{psr.name}_{backend}_log10_t2equad': -2.5
    }

    result_simple = noise_simple_var.getN(params_simple)
    result_full = noise_full_var.getN(params_full)

    np.testing.assert_allclose(result_simple, result_full, rtol=1e-10)


@pytest.mark.unit
def test_makegp_ecorr_simple_consistency():
    """Test that makegp_ecorr_simple gives same results as makegp_ecorr
    when using a single backend with scale=1.0, enterprise=False"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    # Test with fixed parameters
    noisedict_simple = {
        f'{psr.name}_log10_ecorr': -2.5
    }
    noisedict_full = {
        f'{psr.name}_{backend}_log10_ecorr': -2.5
    }

    gp_simple = signals.makegp_ecorr_simple(psr, noisedict=noisedict_simple)
    gp_full = signals.makegp_ecorr(psr, noisedict=noisedict_full, scale=1.0, enterprise=False)

    # Both should be ConstantGP
    assert isinstance(gp_simple, matrix.ConstantGP)
    assert isinstance(gp_full, matrix.ConstantGP)

    # Check that F matrices match
    np.testing.assert_array_equal(gp_simple.F, gp_full.F)

    # Check that Phi values match
    np.testing.assert_allclose(gp_simple.Phi.N, gp_full.Phi.N, rtol=1e-10)


@pytest.mark.unit
def test_makegp_ecorr_simple_variable():
    """Test makegp_ecorr_simple with variable parameters"""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)

    # This should trigger the variable path and reveal the bug on line 180
    try:
        gp = signals.makegp_ecorr_simple(psr, noisedict={})

        # Should be VariableGP
        assert isinstance(gp, matrix.VariableGP)

        # Check parameters - this will fail if there's a bug
        assert hasattr(gp.Phi, 'params')
        params = gp.Phi.params
        assert f'{psr.name}_log10_ecorr' in params

        # Try to evaluate it
        test_params = {f'{psr.name}_log10_ecorr': -2.5}
        phi = gp.Phi.getN(test_params)

        # Should be all equal values
        expected_phi = 10.0 ** (2.0 * -2.5)
        np.testing.assert_allclose(phi, expected_phi, rtol=1e-10)

    except NameError as e:
        # If we get NameError about 'Params' not being defined, that confirms the bug
        if 'Params' in str(e):
            pytest.fail(f"Bug found in makegp_ecorr_simple line 180: {e}. Should be 'params' not 'Params'")
        else:
            raise


@pytest.mark.unit
def test_makegp_ecorr_variable_flag():
    """Test makegp_ecorr with variable=True flag.
    This creates a VariableGP even with fixed noisedict, meaning GP coefficients
    are not automatically marginalized over."""
    psr = MockPulsar('J0437-4715', ntoas=50, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    noisedict = {
        f'{psr.name}_{backend}_log10_ecorr': -2.5
    }

    # With variable=False (default), should get ConstantGP
    gp_constant = signals.makegp_ecorr(psr, noisedict=noisedict, variable=False)
    assert isinstance(gp_constant, matrix.ConstantGP)

    # With variable=True, should get VariableGP even though noisedict is complete
    gp_variable = signals.makegp_ecorr(psr, noisedict=noisedict, variable=True)
    assert isinstance(gp_variable, matrix.VariableGP)

    # The VariableGP should have an empty params list (since noisedict is fixed)
    assert hasattr(gp_variable.Phi, 'params')
    assert gp_variable.Phi.params == []

    # Check that it has the expected attributes for GP coefficients
    assert hasattr(gp_variable, 'index')
    assert hasattr(gp_variable, 'name')
    assert hasattr(gp_variable, 'pos')
    assert hasattr(gp_variable, 'gpname')
    assert hasattr(gp_variable, 'gpcommon')

    # Check the index contains the coefficients parameter
    assert len(gp_variable.index) == 1
    coeff_key = list(gp_variable.index.keys())[0]
    assert 'coefficients' in coeff_key
    assert gp_variable.name == psr.name
    assert gp_variable.gpname == 'ecorrGP'

    # The phi values should be the same
    phi_constant = gp_constant.Phi.N
    phi_variable = gp_variable.Phi.getN({})  # empty params dict
    np.testing.assert_allclose(phi_constant, phi_variable, rtol=1e-10)


@pytest.mark.unit
def test_ecorr_gp_vs_noise_equivalence():
    """Test that ECORR as a GP component gives same result as ECORR in noise matrix.

    Two approaches should be equivalent:
    1. makenoise_measurement(ecorr=False) + makegp_ecorr as separate GP
    2. makenoise_measurement(ecorr=True) with ECORR in noise matrix

    Both should produce the same effective noise covariance matrix.
    Tests both enterprise=True and enterprise=False modes.
    """
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    # Create a mock pulsar with single backend
    psr = MockPulsar('J0437-4715', ntoas=100, nbackends=1)
    backend = sorted(set(psr.backend_flags))[0]

    # Fixed noise parameters
    noisedict = {
        f'{psr.name}_{backend}_efac': 1.2,
        f'{psr.name}_{backend}_log10_t2equad': -2.5,
        f'{psr.name}_{backend}_log10_ecorr': -2.5
    }

    # Test with enterprise=True (only multi-TOA epochs)
    # Approach 1: ECORR as separate GP component
    noise_no_ecorr = ds.makenoise_measurement(psr, noisedict=noisedict, ecorr=False)
    gp_ecorr = ds.makegp_ecorr(psr, noisedict=noisedict, enterprise=True)

    likelihood1 = ds.PulsarLikelihood([psr.residuals, noise_no_ecorr, gp_ecorr])

    # Approach 2: ECORR in noise matrix (Sherman-Morrison form)
    noise_with_ecorr = ds.makenoise_measurement(psr, noisedict=noisedict, ecorr=True, enterprise=True)

    likelihood2 = ds.PulsarLikelihood([psr.residuals, noise_with_ecorr])

    # Get the top-level N from each likelihood
    N1 = likelihood1.N
    N2 = likelihood2.N

    # Test solve_1d on a random vector
    # The two approaches use different internal representations (Woodbury vs Sherman-Morrison)
    # but should give the same results for solve_1d
    np.random.seed(42)  # For reproducibility
    y = np.random.randn(len(psr.toas))
    Nmy1, logN1 = N1.solve_1d(y)
    Nmy2, logN2 = N2.solve_1d(y)

    # Both should give the same result
    np.testing.assert_allclose(Nmy1, Nmy2, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(logN1, logN2, rtol=1e-10, atol=1e-10)

    # Test with enterprise=False (all epochs including single-TOA)
    # Approach 1: ECORR as separate GP component
    noise_no_ecorr_all = ds.makenoise_measurement(psr, noisedict=noisedict, ecorr=False)
    gp_ecorr_all = ds.makegp_ecorr(psr, noisedict=noisedict, enterprise=False)

    likelihood3 = ds.PulsarLikelihood([psr.residuals, noise_no_ecorr_all, gp_ecorr_all])

    # Approach 2: ECORR in noise matrix (Sherman-Morrison form)
    noise_with_ecorr_all = ds.makenoise_measurement(psr, noisedict=noisedict, ecorr=True, enterprise=False)

    likelihood4 = ds.PulsarLikelihood([psr.residuals, noise_with_ecorr_all])

    # Get the top-level N from each likelihood
    N3 = likelihood3.N
    N4 = likelihood4.N

    # Test solve_1d on a random vector
    np.random.seed(43)  # Different seed for second test
    y2 = np.random.randn(len(psr.toas))
    Nmy3, logN3 = N3.solve_1d(y2)
    Nmy4, logN4 = N4.solve_1d(y2)

    # Both should give the same result
    np.testing.assert_allclose(Nmy3, Nmy4, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(logN3, logN4, rtol=1e-10, atol=1e-10)


"""
Pytest tests for makegp_ecorr bug fixes (PR #100, Issue #99)
"""



@pytest.fixture
def data_dir():
    """Path to test data directory"""
    return Path(__file__).parent / "data"

@pytest.mark.unit
def test_single_backend_pulsar(data_dir):
    """Test single-backend bin bug fix"""
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    psr = ds.Pulsar.read_feather(data_dir / "single_backend_pulsar.feather")

    # Should not raise an error
    gp_ecorr = ds.makegp_ecorr(psr, psr.noisedict)

    # Should have non-empty F matrix
    assert hasattr(gp_ecorr, 'F')
    assert gp_ecorr.F.shape[0] == len(psr.toas)

@pytest.mark.unit
def test_empty_epoch_pulsar(data_dir):
    """Test empty-epoch bug fix"""
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    psr = ds.Pulsar.read_feather(data_dir / "empty_epoch_pulsar.feather")

    # Should not raise an error even with backend having no simultaneous TOAs
    gp_ecorr = ds.makegp_ecorr(psr, psr.noisedict)

    assert hasattr(gp_ecorr, 'F')

@pytest.mark.unit
def test_multi_backend_pulsar(data_dir):
    """Test multi-backend control case"""
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    psr = ds.Pulsar.read_feather(data_dir / "multi_backend_pulsar.feather")

    # Should work correctly
    gp_ecorr = ds.makegp_ecorr(psr, psr.noisedict)

    assert hasattr(gp_ecorr, 'F')
    assert gp_ecorr.F.shape[0] == len(psr.toas)

@pytest.mark.unit
def test_missing_bin_zero_bug(data_dir):
    """Test that makegp_ecorr includes bin 0 when mask has no zeros.
    Fixes issue #99.
    """
    try:
        import discovery as ds
    except ImportError:
        pytest.skip("discovery package not installed")

    psr = ds.Pulsar.read_feather(data_dir / "single_backend_pulsar.feather")

    # For a single-backend pulsar, the mask should have no zeros
    backend_flags = ds.selection_backend_flags(psr)
    backends = [b for b in sorted(set(backend_flags)) if b != '']
    masks = [np.array(backend_flags == backend) for backend in backends]

    # Verify this is a single-backend case where mask has no zeros
    assert len(backends) == 1, "This test requires a single-backend pulsar"
    mask = masks[0]
    assert np.all(mask), "Mask should have no zeros for single-backend pulsar"

    # Check what bins quantize returns
    masked_toas = psr.toas * mask
    bins = ds.quantize(masked_toas)
    unique_bins = np.unique(bins)
    bins_max = bins.max()
    print(f"Unique bins from quantize: {unique_bins}")
    print(f"bins.max(): {bins_max}")

    # Expected behavior: if bins go from 0 to bins_max, we should have bins_max + 1 columns
    # Buggy behavior: range(1, bins.max() + 1) gives only bins_max columns (missing bin 0)
    if 0 in unique_bins and bins_max > 0:
        # If bin 0 exists and there are other bins, we should have bins_max + 1 columns
        expected_num_columns = bins_max + 1
        # But the buggy code using range(1, bins.max() + 1) would only give bins_max columns
        buggy_num_columns = bins_max
    else:
        # If bin 0 doesn't exist or there's only one bin, the current code might be correct
        expected_num_columns = len(unique_bins)
        buggy_num_columns = expected_num_columns

    # Call makegp_ecorr
    gp_ecorr = ds.makegp_ecorr(psr, psr.noisedict)

    # Verify the F matrix has the correct number of columns
    assert hasattr(gp_ecorr, 'F'), "GP should have F attribute"
    print(f"gp_ecorr.F.shape: {gp_ecorr.F.shape}")
    print(f"Expected number of columns (including bin 0): {expected_num_columns}")
    print(f"Buggy number of columns (missing bin 0): {buggy_num_columns}")
    print(f"Actual number of columns: {gp_ecorr.F.shape[1]}")

    # This assertion will fail if bin 0 is being skipped
    # If 0 is in unique_bins and bins_max > 0, we should have bins_max + 1 columns
    if 0 in unique_bins and bins_max > 0:
        assert gp_ecorr.F.shape[1] == expected_num_columns, (
            f"F matrix should have {expected_num_columns} columns (bins 0 through {bins_max}), "
            f"but got {gp_ecorr.F.shape[1]}. This indicates bin 0 is being skipped by "
            f"range(1, bins.max() + 1)."
        )