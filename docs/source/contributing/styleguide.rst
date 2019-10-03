Styleguide
==========

Your contribution should fulfill the following criteria:

- PEP8 compliant and black formatted (we check this automatically).
    We make this
    such a hard requirement because it's boring and we don't want to bother about
    it in code reviews. Not because we think that all PEP8 compliant code is
    automatically good. Watch `this video <https://www.youtube.com/watch?v=wf-BqAjZb8M>`_
    if you haven't seen it yet.
- All functions have a `Google style <https://tinyurl.com/mxams9k>`_ docstring
    that describes all arguments and outputs. For arrays, please document how
    many dimensions and which shape they have. Look around in the code to find
    examples if you are in doubt.
- Unit tests.
    If you write a small helper whose interface might change during refactoring,
    it is sufficient if the function that calls it is tested.
    But all functions that are exposed to the user must have unit tests.
- Use ``pathlib`` for all file paths operations.
- Functions have no side effect.
    If you modify a mutable argument, make a copy at the beginning of the function.
- Prefer a functional style over object oriented programming.
    Unless you have very good reasons for writing a class, we prefer you don't do
    it. You might want to watch `this <https://www.youtube.com/watch?v=o9pEzgHorH0>`_
- Don't use global variables
- Deep modules.
    This is a term coined by
    `John Ousterhout <https://www.youtube.com/watch?v=bmSAYlu0NcY>`_. A deep module
    is a module that has just one public function. This function calls the private
    functions (i.e. functions that start with an underscore) defined further down
    in the module and reads almost like a table of contents to the whole module.
- Never import a private function in another module

