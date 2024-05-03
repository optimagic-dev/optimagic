(eeppytrees)=

# EEP-01: Static typing

```{eval-rst}
+------------+------------------------------------------------------------------+
| Author     | `Janos Gabler <https://github.com/janosg>`_                      |
+------------+------------------------------------------------------------------+
| Status     | Draft                                                            |
+------------+------------------------------------------------------------------+
| Type       | Standards Track                                                  |
+------------+------------------------------------------------------------------+
| Created    | 2024-05-02                                                       |
+------------+------------------------------------------------------------------+
| Resolution |                                                                  |
+------------+------------------------------------------------------------------+
```

## Abstract

This enhancement proposal explains how we want to adopt static typing in estimagic. The
overarching goals of the proposal are the folloing:

- More robust code due to static type checking
- Better readability of code due to type hints
- Better discoverability and autocomplete for users of estimagic

Achieving these goals requires more than adding type hints. Estimagic is currently
mostly [stringly typed](https://wiki.c2.com/?StringlyTyped) and full of dictionaries
with a fixed set of required keys (e.g.
[constraints](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_constraints.html),
[option dictionaries](https://estimagic.readthedocs.io/en/latest/how_to_guides/optimization/how_to_specify_algorithm_and_algo_options.html),
etc.).

This enhancement proposal outlines how we can accomodate the changes needed to reap the
benefits of static typing without breaking users' code.

## Motivation and ressources

## Changes in public functions

### Constraints

### Option dictionaries

### Algorithm selection

### Benchmark problems and results

### Least-squares and likelihood problems

## Internal changes

### Internal algorithm interface

## Summary of design philosophy

## Changes in documentation

- No type hints in docstrings anymore
- Only show new recommended ways of doing things, not deprecated ones

## Breaking changes

## Deprecations
