"""Backward-compat re-export. The matrix→metamath patch logic now lives in
`discovery._kernel_switch` so `ds.config(kernels='metamath')` can apply it
persistently. The test harness uses the context-manager form."""
from discovery._kernel_switch import patched_kernels as metamatrix_patch

__all__ = ["metamatrix_patch"]
