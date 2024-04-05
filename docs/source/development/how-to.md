# How to contribute

Contributions of all forms and sizes are welcome and highly appreciated! Anything
ranging from updates to the documentation and small extensions to implementing new
features. For substantial changes, please get in touch with us beforehand. This enables
us to discuss your proposals and potentially guide the development process from the
start. You can initiate a conversation by posting an issue or directly reaching out to
[janosg](https://github.com/janosg) via email.

To familiarize yourself with the codebase, you can check out our
[issue tracker](https://github.com/OpenSourceEconomics/estimagic/issues) for some
immediate and clearly defined tasks.

Assuming you have settled on contributing to the project, we advise reviewing the
{ref}`style_guide` available on the following page to ensure consistency with the
project's coding standards.

To begin contributing, start by cloning the
[repository](https://github.com/OpenSourceEconomics/estimagic/) to your local machine.
This will create a copy of estimagic's repository where you have local write access.
You'll implement all changes and fixes on your local estimagic copy before opening a
Pull Request (PR). With a PR you propose your changes to be merged into the project's
main branch. Regular contributors receive push access to unprotected branches,
simplifying the contribution process.

Here's a step-by-step guide for making contributions via PR, adhering to the estimagic
style guide:

1. Clone the repository to your disk and set up your project environment using conda.

   Open the terminal and execute the following commands from the root directory of your
   local estimagic repository

   ```bash
   $ conda env create -f environment.yml
   ```

   ```bash
   $ conda activate estimagic
   ```

   This automatically installs estimagic in editable mode.

1. Create a new local branch for your work with

   ```bash
   $ git checkout -b YOUR_BRANCH
   ```

   This assumes that you are starting from the main branch.

1. Implement your fix or feature. To stage your changes for commit, type

   ```bash
   $ git add src/estimagic/YOUR_FILE.py
   ```

   To learn more about git, how to add and commit changes, have a look at these
   [online materials](https://effective-programming-practices.vercel.app/git/staging/objectives_materials.html).

1. We validate contributions in two ways. First, we employ a comprehensive test suite to
   check if new implementations are compatible with estimagic's existing codebase.
   Second, we use
   [pre-commit hooks](https://effective-programming-practices.vercel.app/git/pre_commits/objectives_materials.html)
   to ensure contributions meet our quality standards and adhere to our stylistic
   guidelines.

   Run the test suit with

   ```bash
   $ pytest
   ```

   Look at the summary report and fix any errors that pop up.

   To enable pre-commit hooks for linting and stylistic error-checking, type

   ```bash
   $ pre-commit install
   ```

   With pre-commit installed, linters are executed before each commit. A commit is
   rejected if any of the checks fails. Note that some linters fix the errors
   automatically by modifying the code in-place. Restage the respective files via
   `git add src/estimagic/YOUR_FILE.py`.

   You can also run the linters on the entire project via

   ```bash
   $ pre-commit run -a
   ```

1. If you have updated the documentation, check if it builds correctly. To do so, go to
   the root directory of your local estimagic repo and navigate to the `docs` folder.
   There, you need to create the `estimagic-docs` environment and activate it

   ```bash
   $ conda env create -f rtd_environment.yml
   ```

   ```bash
   $ conda activate estimagic-docs
   ```

   Inside the `docs` folder, type

   ```bash
   $ make html
   ```

   and sphinx automatically builds the documentation locally. You can view the built
   html documentation in your browser (e.g. Google Chrome) via

   ```bash
   $ google-chrome build/html/index.html
   ```

   and check if everything looks fine. Note that the command above opens the index page,
   from where you can navigate to other sections of the estimagic documentation. If you
   wish to directly open a specific page, you can type

   ```bash
   $ google-chrome build/html/explanations/optimization/why_optimization_is_hard.html
   ```

1. Once all tests and pre-commit hooks pass locally, push your branch to the remote
   estimagic repository

   ```bash
   $ git push --set-upstream origin YOUR_BRANCH
   ```

   and create a pull request through the GitHub interface: Go to estimagic's Github
   page. A banner will be displayed asking you whether you would like to create a pull
   request. Click on the link.

   Follow the instructions of the estimagic
   [PR template](https://github.com/OpenSourceEconomics/estimagic/blob/main/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md)
   to describe your contribution, the problem you want to solve, and your proposed
   solution.

   Opening a PR starts a complete run of the test suite on a Continuous Integration (CI)
   server. The status of the CI run is shown on your PR page. If necessary, make
   modifications to your code and reiterate until the tests pass on the remote machine.

1. Ask one of the main contributors to review your changes. If they have any remarks or
   suggestions, address them and commit your modifications.

1. Once you're PR is approved, one of the main contributors will merge it into
   estimagic's main branch.
