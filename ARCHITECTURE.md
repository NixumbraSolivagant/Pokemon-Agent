# Project Architecture

## Runtime Layers

1. `main.py` contains the Kaggle-compatible single-file policy entrypoint.
2. `cg/` wraps the official simulator and native search libraries.
3. `local_eval/` runs submissions in isolated processes and produces reports.
4. `tools/` builds, evaluates, searches, and promotes candidate submissions.
5. `tests/` protects evaluator behavior and build/discovery contracts.

The submission stays single-file because Kaggle selects a callable from
`main.py`. Supporting automation should remain modular and must not move
runtime-only imports into the generated submission unless the packager also
includes those modules.

## Tool Dependencies

Tool modules should follow this dependency direction:

```text
build_models.py  deck_rules.py
        \          /
         build_submission.py
                  |
      generation and discovery tools
```

- `tools/build_models.py` owns build configuration data only.
- `tools/deck_rules.py` owns deck validation and deterministic mutations.
- `tools/build_submission.py` owns archive transformation and search-wrapper
  rendering.
- Discovery and iteration tools may depend on the model and rule modules, but
  should import the builder only when they actually build an archive.

`tools/build_submission.py` re-exports the model and deck-rule names for
backward compatibility. New code should import them from their owning modules.

## Agent Maintenance

- Keep card identifiers and game-specific heuristics inside `main.py` until a
  source-to-single-file build step is introduced.
- Route public-agent failures through `_fallback_action()` so legality handling
  has one implementation.
- Resolve bundled files relative to `__file__`; do not rely on the caller's
  working directory.
- Treat the injected search wrapper as generated code. Changes to its behavior
  belong in `tools/build_submission.py` and require archive-level tests.

## Validation

Run the focused validation sequence after structural changes:

```bash
python3 -m compileall -q main.py cg local_eval tools tests
git diff --check
python3 -m pytest -q
```
