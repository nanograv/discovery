# Design notes

Durable design documents for the **metamatrix** graph backend and
**single-precision** likelihood work. These complement the Sphinx guides; they
are not session handoffs or lab harnesses.

## Product docs (start here)

| Doc | Audience |
|---|---|
| [Metamatrix user guide](../metamatrix.md) | Model builders |
| [Metamatrix developer guide](../metamatrix_dev.md) | Contributors / maintainers |
| [Single-precision tutorial](../advanced/single_precision) | float32 / reference+delta usage |

## Design archive (this tree)

### Metamatrix

| File | Role |
|---|---|
| [metamatrix/architecture.md](metamatrix/architecture.md) | Why graphs; house rules; porting guidance |
| [metamatrix/parity_coverage.md](metamatrix/parity_coverage.md) | Constructor → parity-test map |
| [metamatrix/deletion_checklist.md](metamatrix/deletion_checklist.md) | Remaining legacy-path cutover steps |

### Single precision

| File | Role |
|---|---|
| [single_precision/README.md](single_precision/README.md) | Overview + glossary |
| [single_precision/adr/](single_precision/adr/) | Architecture decision records |
| [single_precision/research_refdelta.md](single_precision/research_refdelta.md) | Single-level ΔlogL math |
| [single_precision/research_nested.md](single_precision/research_nested.md) | Nested/fused HD ΔlogL math |
| [single_precision/future.md](single_precision/future.md) | Deferred ideas |

## What does *not* live here

Process notes, session handoffs, experiment “finding” tables, branch-diff
snapshots, and offline harness scripts were removed from the tree (they remain
in git history). Official regression coverage is under `tests/metamatrix/` and
`tests/single_precision/`.
