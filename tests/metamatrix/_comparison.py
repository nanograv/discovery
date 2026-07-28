"""Scale-aware comparison helpers for matrix.py-vs-metamath parity tests.

Tolerance picked by `kind`:
  - 'logL'      : log-likelihood scalars (can be O(1e6)). atol scales with |old|.
  - 'residuals' : seconds. atol fixed at 1e-16 (sub-clock).
  - 'coeffs'    : fourier amplitudes / latent vectors. dimensionless-ish.
  - 'matrix'    : cholesky factors or similar.
"""

import numpy as np


def assert_close(new, old, *, kind, name=""):
    new = np.asarray(new)
    old = np.asarray(old)

    if kind == "logL":
        scale = max(1.0, float(np.max(np.abs(old))))
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-8 * scale,
                                   err_msg=f"{name} logL diverged")
    elif kind == "residuals":
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-16,
                                   err_msg=f"{name} residuals diverged")
    elif kind == "coeffs":
        scale = max(1.0, float(np.max(np.abs(old))))
        np.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-10 * scale,
                                   err_msg=f"{name} coefficients diverged")
    elif kind == "matrix":
        # cho factors of Sigma can be O(1e7+); roundoff accumulates in the solve
        scale = max(1.0, float(np.max(np.abs(old))))
        np.testing.assert_allclose(new, old, rtol=1e-8, atol=1e-10 * scale,
                                   err_msg=f"{name} matrix diverged")
    else:
        raise ValueError(f"unknown comparison kind: {kind!r}")


def assert_params_equal(new_fn, old_fn, name=""):
    new_p, old_p = set(new_fn.params), set(old_fn.params)
    only_old = old_p - new_p
    only_new = new_p - old_p
    assert not (only_old or only_new), (
        f"{name} param drift — only old: {sorted(only_old)}, "
        f"only new: {sorted(only_new)}"
    )
