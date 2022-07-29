
.. _style_guide:

# Styleguide

Your contribution should fulfill the criteria provided below.

## Styleguide for the codebase

- Functions have no side effect.
    If you modify a mutable argument, make a copy at the beginning of the function.
- Deep modules.
    This is a term coined by
    [John Ousterhout] (https://www.youtube.com/watch?v=bmSAYlu0NcY)`_. A deep module
    is a module that has just one public function. This function calls the private
    functions (i.e. functions that start with an underscore) defined further down
    in the module and reads almost like a table of contents to the whole module.
- Use good names for functions and variables
    *"You should name a variable using the same care with which you name a first-born
    child."*, Robert C. Martin, Clean Code: A Handbook of Agile Software Craftsmanship.

    A bit more concretely, this means:

    The length of a variable name should be proportional to its scope.
    In a list comprehension or short loop, i might be an acceptable name for
    the running variable, but variables that are used at many different
    places should have descriptive names.

    The name of variables should reflect the content or meaning of the
    variable and not only the type. Names like ``dict_list`` would not
    have been a good name for the ``constraints``.

    Function names should contain a verb. Moreover, the length of a
    function name is typically inversely proportional to its scope. The public
    functions like ``maximize`` and ``minimize`` can have very short names.
    At a lower level of abstraction you typically need more words to describe
    what a function does.
- Never import a private function in another module
    By following this strictly, you can be sure that you can rename or refactor
    private functions without looking at other modules. Of course it is also not
    a solution to copy paste the function! If you would like to import a function
    that starts with an underscore, rename it.
- All functions have a [Google style] (https://tinyurl.com/mxams9k)`_ docstring
    The docstring describes all arguments and outputs. For arrays, please document
    how many dimensions and what shape they have. Look around in the code to find
    examples if you are in doubt. Example:

   ```python

        def ordered_logit(formula, data, dashboard=False):
            """Estimate an ordered probit model with maximum likelihood.

            Args:
                formula (str): A patsy formula.
                data (str): A pandas DataFrame.
                dashboard (bool): Switch on the dashboard.

            Returns:
                res: optimization result.

            """
            pass
    ```
    In particular each docstring should start with a one liner that describes
    very concisely what the function does. The one liner should be in
    imperative mode, i.e. not "This function does" ..." , but "Do ..."
    and end with a period.

- Unit tests.
    If you write a small helper whose interface might change during refactoring,
    it is sufficient if the function that calls it is tested.
    But all functions that are exposed to the user must have unit tests.
- PEP8 compliant and black formatted (we check this automatically).
    We make this such a hard requirement because it's boring and we don't
    want to bother about it in code reviews. Not because we think that all
    PEP8 compliant code is automatically good.
    Watch [this video] (https://www.youtube.com/watch?v=wf-BqAjZb8M)`_
    if you haven't seen it yet.
- Use ``pathlib`` for all file paths operations.
    You can find the pathlib documentation
    [here] (https://docs.python.org/3/library/pathlib.html)`_
- Object serialization.
    Pickling and unpickling of DataFrames should be done with ``pd.read_pickle``
    and ``pd.to_pickle``.
- We prefer a functional style over object oriented programming.
    Unless you have very good reasons for writing a class, we prefer you don't do
    it. You might want to watch [this] (https://www.youtube.com/watch?v=o9pEzgHorH0)`_
- Don't use global variables unless absolutely necessary
    Exceptions are global variables from a config file that replace magic numbers.
    Never use mutable global variables!

## Styleguide for the documentation

- General.
    The documentation is rendered with [Sphinx] <https://www.sphinx-doc.org/en/master/>`_
    and  written in **Markedly Structured Text.** How-to guides are usually Jupyter notebooks.

- Purpose of documents.
    Our documentation is inspired by the [system] (https://documentation.divio.com/)`_
    developed by Daniele Procida.

      - How-to guides are problem-oriented and show how to achieved specific tasks.
      - Explanations contain information on theoretical
        concepts underlying estimagic, such as numerical differentiation and
        moment-based estimation.
      - The API Reference section contains auto-generated API reference
        documentation and provides additional details about the implementation.

- Headings.
    Only the first letter of a title is capitalized.

- Format.
    The code formatting in .md files is ensured by blacken-docs.
