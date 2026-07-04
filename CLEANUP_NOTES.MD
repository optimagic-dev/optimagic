Here we collect task to work on after the main refactoring of constraints is done.


## Bug Fixes



## Improvements

- Check whether there are more cases like diagonal covariance matrices or 1x1 covariance matrices that can actually be covered without A reparametrization. A good candidate are probability constraints with 2 or less entries. I haven't checked if this is already exploited.
- Update and extend the documentation (how constraints are implemented)
- Do we really need the flat param names for error messages in ResolutionContext if we have the source and can apply selector to params?



## Clean UP

- Remove all mentions of the constraints refactoring (e.g. in test_constraint_pipeline_invariants)
- The IntArray and FloatArray types in optimagic.parameters.constraints.types should actually be defined in optimagic.typing and used throughout.
- Split up constraints.py into one file per constraint type that also contains the transformations
