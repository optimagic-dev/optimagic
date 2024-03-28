# How to contribute

Contributions are always welcome and highly appreciated! Anything ranging from small
extensions of the documentation to implementing new features. Of course, the bigger the
change the more you are encouraged to reach out to us in advance. That way we can start
discussing your ideas before you actually start implementing them. You can post an issue
or contact [janosg](https://github.com/janosg) via email.

To get acquainted with the code base, you can check out our
[issue tracker](https://github.com/OpenSourceEconomics/estimagic/issues) for some
immediate and clearly defined tasks.

Assuming you have settled on contributing a small fix to the project, please read the
{ref}`style_guide` on the next page before you continue.

Next, clone the [repository](https://github.com/OpenSourceEconomics/estimagic/) to your
local machine. This will create a copy of the repository where you have local write
access. Your fix will be implemented in your copy. After that, you will open a Pull
Request (PR). With a PR you propose your changes to be merged into the project. If you
plan to become a regular contributor, we can give you push access to unprotected
branches, which makes the process more convenient for you.

Below is a step-by-step guide on how to implement your changes via a PR while adhering
to estimagic's style guide.

1. Clone the repository to your disk. Set up the project environment with conda. This
   will automatically install your local version of estimagic in editable mode.

   On your computer, open the terminal in the root of your local estimagic repo and
   enter the following commands:

   ```bash
   $ conda env create -f environment.yml
   ```

   ```bash
   $ conda activate estimagic
   ```

   ```bash
   $ pip install -e .
   ```

1. Implement the fix or new feature. To stage your changes for commit, type

   ```bash
   $ git add src/estimagic/YOUR_FILE.py
   ```

   To learn more about git and how to add/commit your local changes, have a look at the
   materials
   [here](https://effective-programming-practices.vercel.app/git/staging/objectives_materials.html).

1. We validate contributions in two ways. First, we use a comprehensive test suite to
   check if your changes are compatible with estimagic. Second, we use
   [pre-commit hooks](https://effective-programming-practices.vercel.app/git/pre_commits/objectives_materials.html)
   to fix stylistic errors in code and documentation.

   You can run the test suit via

   ```bash
   $ pytest
   ```

   Look at the summary report and fix any errors that pop up.

   To enable stylistic error-checking and linting via pre-commit, type

   ```bash
   $ pre-commit install
   ```

   With pre-commit installed, linters are executed before each commit and the commit is
   rejected if any of the checks fails. Note that some linters fix errors directly by
   automatically modifying the code. Restage the respective files via
   `git add src/estimagic/YOUR_FILE.py`.

   You can also run the linters manually via

   ```bash
   $ pre-commit run -a
   ```

1. If you have made changes to the estimagic documentation, check if the it builds
   correctly. To do so, go to the root directory of your local estimagic repo and
   navigate to the `docs` folder. There, you need to create the `estimagic-docs`
   environment and activate it

   ```bash
   $ conda env create -f rtd_environment.yml
   ```

   ```bash
   $ conda activate estimagic-docs`
   ```

   Still in the `docs` folder, now build the sphinx documentation pages by typing

   ```bash
   $ make html
   ```

   and sphinx automatically builds the documentation locally. You can then open the html
   pages in you browser (e.g. Google Chrome) via

   ```bash
   $ google-chrome build/html/index.html
   ```

   and check if everything looks fine. Note that the command above opens the index page,
   from where you can navigate to other sections of the estimagic documentation. If you
   wish to directly open a specific documentation page, you can type

   ```bash
   $ google-chrome build/html/explanations/optimization/why_optimization_is_hard.html
   ```

1. If all tests and pre-commit hooks have passed locally, you are ready to push your
   changes to a new branch on the remote estimagic repository.

   Go to estimagic's Github page. A banner will be displayed asking you whether you
   would like to create a pull request. Click on the link and follow the instructions of
   the PR template. Fill out the PR form to inform everyone what you have been working
   on, what you have been trying to accomplish, and what you have done to achieve that.

   Opening a PR starts a complete run of the test suite on a Continuous Integration (CI)
   server. The status of the CI run is shown on your PR page. If necessary, make
   modifications to your code and reiterate until the tests pass on the remote machine.

1. Ask one of the main contributors to review your changes. If they have any remarks or
   suggestions, address them and commit your modifications.

1. Once you're PR is approved, one your the main contributors will merge it into
   estimagic's main branch.
