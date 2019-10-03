How To Contribute
=================

Contributions are always welcome. Everything ranging from small extensions of the
documentation to implementing new features is appreciated. Of course, the
bigger the change the more it is necessary to reach out to us in advance for a
discussion. You can post an issue or contact janosg via email.

To get acquainted with the code base, you can also check out our `issue tracker
<https://github.com/OpenSourceEconomics/estimagic/issues>`_ for some immediate and clearly
defined tasks.



1. Assuming you have settled on contributing a small fix to the project, fork the
   `repository <https://github.com/OpenSourceEconomics/estimagic/>`_. This will create a
   copy of the repository where you have write access. Your fix will be implemented in
   your copy. After that, you will start a pull request (PR) which means a proposal to
   merge your changes into the project. If you plan to become a regular contributor
   we can give you push access to unprotected branches, which makes the process more
   convenient for you.

2. Clone the repository to your disk. Set up the environment of the project with conda
   and the ``environment.yml``. Implement the fix.

3. We validate contributions in three ways. First, we have a test suite to check the
   implementation of estimagic. Second, we correct for stylistic errors in code and
   documentation using linters. Third, we test whether the documentation builds
   successfully.

   You can run all checks with ``tox`` by running

   .. code-block:: bash

       $ tox

   This will run the complete test suite. To run only a subset of the suite you can use
   the environments, ``pytest``, ``linting`` and ``sphinx``, with the ``-e`` flag of
   tox.

   Correct any errors displayed in the terminal.

   To correct stylistic errors, you can also install the linters as a pre-commit with

   .. code-block:: bash

       $ pre-commit install

   Then, all the linters are executed before each commit and the commit is aborted if
   one of the check fails. You can also manually run the linters with

   .. code-block:: bash

       $ pre-commit run -a

4. If the tests pass, push your changes to your repository. Go to the Github page of
   your fork. A banner will be displayed asking you whether you would like to create a
   PR. Follow the link and the instructions of the PR template. Fill out the PR form to
   inform everyone else on what you are trying to accomplish and how you did it.

   The PR also starts a complete run of the test suite on a continuous integration
   server. The status of the tests is shown in the PR. Reiterate on your changes until
   the tests pass on the remote machine.

5. Ask one of the main contributors to review your changes. Include their remarks in
   your changes.

6. The final PR will be merged by one of the main contributors.
