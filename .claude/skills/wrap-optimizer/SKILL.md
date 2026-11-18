---
name: wrap-optimizer
description: Add an optimizer wrapper to optimagic or bring an existing wrapper up to the wrapper standards.
disable-model-invocation: true
---

Wrap an optimizer for optimagic, or bring an existing wrapper up to standard. Both
branches run the same steps; a paragraph marked **New** or **Existing** applies to
one branch only.

The rules are in two docs pages. Each step names the sections to read before you do
it:

- `docs/source/how_to/how_to_add_optimizers.md`, part "Standards for wrappers"
- `docs/source/how_to/how_to_document_optimizers.md`

Reference implementations: `src/optimagic/optimizers/fides.py` with
`tests/optimagic/optimizers/test_fides_wrappers.py` (flat options, objects from the
library, a changed default) and `nag_optimizers.py` with `test_nag_wrappers.py` (option
groups, converted deprecations).

The spine of the work is the **option table**: one row per option of the library.
Every step after the audit is checked against its rows.

## 1. Audit the library

Work against the newest release. **New:** add the dependency as described in
"Integrate the wrapper into optimagic". **Existing:** raise the minimal version in
`pyproject.toml` to the newest release, run `pixi install`, and read the changelog
between the old and the new minimal version.

Read the papers and the documentation, then establish every fact from the installed
source: documentation goes stale. For each option, follow it from the entry point to
the lines that use it and note its default, the range and type it accepts, and its
effect. Note also whether the library treats an absent option differently from one
passed as `None`, and whether a default depends on the problem. Where the source
leaves a doubt, run the library directly on a small problem and look.

**Existing:** also list every defect of the current wrapper: options that are never
passed, passed under a wrong key, typed so that they cannot be used, results read
from the wrong field. Check whether `replace_and_warn_about_deprecated_algo_options`
in `src/optimagic/deprecations.py` renames an option of this wrapper before it
arrives.

Done when every argument of the library's entry point, every key of its option
dictionaries and every attribute of its configuration objects has a row, and for each
row you can name the function in the library's source where the option takes effect.

## 2. Draft the option table

Read "The option table", "Which options to expose", "Keep the library's grouping",
"Names of options" and "Defaults".

Fill every column of every row, and list the options that are not exposed with their
reasons.

A default that deviates from the library needs **evidence**, and so does adopting a
library default that the current wrapper deviates from. See
[Evidence for defaults](#evidence-for-defaults). The harmonized stopping and
convergence options are the exception: they take the optimagic values, and the
evidence to collect is which criterion ends the runs.

Done when every row has a name, a group, a default, the name it replaces and a planned
test, and every default that differs from the library's has a benchmark result or
"harmonized" next to it.

## 3. Checkpoint: the option table

Present to the user, in plain language with examples from this library:

- the option table and the list of options that are not exposed,
- the defaults that deviate or change, with the evidence,
- **Existing:** the defects found and the deprecations the table implies,
- whether a tutorial is due (section "Tutorials"),
- every open question as a recommendation with its alternative.

Stop here. Continue once the user has approved the table, with their corrections
applied to it.

## 4. Implement

Read "Keep the library's grouping" and "Changing an existing wrapper".

Put the tables that tests and wrapper share into the wrapper module as public
constants: library name to optimagic name, options that are not exposed with reasons,
defaults that deviate with reasons. Build the call to the library from the first
table.

**Existing:** declare every renamed, moved or converted option as a `DeprecatedOption`
and every changed default with `resolve_changed_default`. Announce removals and new
defaults for the minor release after the one that ships the change (see `CHANGES.md`
and the latest git tag for the current version).

Done when every row of the approved table is in the code under its approved name and
default, every defect from the audit is fixed, and
`pixi run tests tests/optimagic/optimization/test_many_algorithms.py -k <algorithm>`
passes.

## 5. Test

Read "Tests".

Probe before you assert: run the tiny problem with two values of the option and write
the assertion for the effect you observed. An option that shows no effect in the probe
is a finding. Go back to the library's source and find out when the option matters.
Put that into its docstring, or, if the option cannot have an effect inside
optimagic, move it to the options that are not exposed and tell the user.

Let forwarding tests run the real solver with a tiny budget behind the spy. They then
also show that the library accepts the value. Let the contract test read the
library's defaults in every context they depend on (noisy or not, few or many
parameters): it then verifies each "depends on the problem" claim of the table.

Done when

- every row has the test planned for it, and every deprecated name and changed default
  has a deprecation test,
- `pixi run tests -m <library>_wrappers` passes in under a minute,
- deleting one entry from the name table makes a contract test fail (try it, then
  restore the entry),
- the default test run still covers the wrapper: `pixi run tests-fast` passes and
  includes the fast tests of this wrapper.

## 6. Document

Read `how_to_document_optimizers.md`.

Done when every option and every attribute of an option group has a docstring with the
three parts of the template, the class docstring lists the unsupported features with
reasons, `pixi run -e docs build-docs` passes, and you have read the rendered entry of
the algorithm from top to bottom.

## 7. Tutorial

Only if a tutorial is due. Read "Tutorials".

Find each example by experiment first: a section stays only if running it shows the
effect it claims. Generate the numbers in the notebook instead of writing them into
the text.

Done when the notebook runs from top to bottom without warnings in under a minute and
is linked from `docs/source/tutorials/index.md` and from the class docstring.

## 8. Finish

- Register the marker and the library as described in "Integrate the wrapper into
  optimagic". Add the entry to `CHANGES.md`.
- Run `pixi run tests` if you changed code outside the wrapper module, otherwise
  `pixi run tests-fast`; then `pixi run -e type-checking ty` and
  `prek run --all-files`.
- Draft an issue for every unsupported feature that a reasonable extension of
  optimagic could support.
- Draft the pull request description: what changed for users, the final option table,
  the evidence for defaults, and the AI-use note the contributor guide asks for
  (`docs/source/development/how_to_contribute.md`, "Using AI tools").

Done when all checks pass and the user has the drafts of the pull request and the
issues to review before anything is published.

## Evidence for defaults

Compare the candidate values on `om.get_benchmark_problems("more_wild")`, with bounds
added if the option only matters with bounds. Keep everything else at the planned
defaults. Report for each candidate:

- the share of problems on which it comes within 1e-4 of the best value any candidate
  found, relative to the improvement over the start value,
- the median number of function evaluations it needs to get there,
- the computation time per iteration, if the option changes the work inside the
  optimizer,
- for convergence options: which criterion ends the runs and on how many problems a
  run stops short of the best value.

The library default stands unless the numbers show that another value is better. Put
the numbers into the checkpoint and later into the pull request description.
