# Deletion checklist: legacy `matrix.py` path

Phases 0–4 of the metamatrix migration are **done**: shared substrate, kernel
factory, `likelihood_metamath`, parity coverage, and carry-overs (mixed-Φ
`CompoundGP`, `CompoundGlobalGP`). The metamath path is the default. This
document is only the remaining **cutover** once maintainers sign off after
external review.

Living summary: [Metamatrix developer guide](../../metamatrix_dev.md) §14.
Constructor coverage: [parity_coverage.md](parity_coverage.md).

## Preconditions

- [ ] External analyses have exercised `kernels='metamath'` on real models.
- [ ] `tests/metamatrix/` green; no open xfail on production constructors.
- [ ] Optional: goldens committed from the matrix route for post-deletion
      regression (`tests/metamatrix/goldens/`, fixed seed).

## Ordered steps

Prefer one commit (or PR) per step so bisect stays easy.

1. **Goldens (optional but recommended)**  
   Generate matrix-route goldens; switch parity suite to golden comparison so
   the matrix oracle is no longer required at runtime.

2. **Remove test monkeypatch path**  
   - Delete `src/discovery/_kernel_switch.py`  
   - Delete `tests/metamatrix/_patch.py`  
   - Drop the `mh_patched` route from `tests/metamatrix/_routes.py`

3. **Collapse the factory**  
   - `_kernels.py`: bind only metamath classes; `set_mode("matrix")` raises  
   - `set_mode("metamath")` becomes a no-op

4. **Delete legacy likelihood / matrix**  
   - Delete `src/discovery/matrix.py`  
   - Delete `src/discovery/likelihood.py`  
   - `git mv likelihood_metamath.py likelihood.py`  
   - Fix `__init__.py` imports (`from .matrix import *` → explicit re-exports
     of utils-owned symbols)

5. **`config(kernels=...)` policy**  
   - `'matrix'` → hard error  
   - `'metamath'` → `DeprecationWarning` for one release, then remove the kwarg  
   - no-arg returns `'metamath'` during the deprecation window

6. **Tests and docs**  
   - Delete or rewrite matrix-only unit tests  
   - Move shared substrate tests (e.g. `make_uind`) under `tests/metamatrix/`  
   - Update notebooks that select paths; refresh [metamatrix.md](../../metamatrix.md)
     / [metamatrix_dev.md](../../metamatrix_dev.md) status language

## Gate

- [ ] Full test suite green  
- [ ] Example / tutorial notebooks build models under the single path  
- [ ] No remaining imports of `likelihood_metamath` or production `matrix` kernel
      constructors  

## Non-goals for this checklist

- Fixing `cglogL` (broken on both routes today; separate work).  
- Multi-ExtSignal cross-term bug in the clogL *cross* form
  ([#137](https://github.com/nanograv/discovery/issues/137)).  
- New single-precision features (see [../single_precision/future.md](../single_precision/future.md)).
