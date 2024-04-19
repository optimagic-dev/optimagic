# How to contribute

## 1. Intro

Contributions of all forms and sizes are welcome and highly appreciated! Anything from
updating the documentation, adding small extensions to implementing new features.

For substantial changes, please get in touch with us beforehand. This allows us to
discuss your ideas and guide the development process right from the start, which may
help clear up any misunderstandings and unecessary work. You can initiate a conversation
by posting an issue on GitHub or reaching out to [janosg](https://github.com/janosg) via
email.

To familiarize yourself with the codebase, we recommed checking out our
[issue tracker](https://github.com/OpenSourceEconomics/estimagic/issues) for some
immediate and clearly defined tasks.

## 2. Before you start

Assuming you have settled on contributing to the project, we advise reviewing the
{ref}`style_guide` (see the next page) to ensure consistency with the project's coding
standards.

We use Pull Requests (PR) to incorporate new features into the estimagic ecosystem.
Contributors work on a local estimagic copy where they can freely modify and extend the
codebase before opening a PR. With a PR you propose your changes to be merged into the
project's main branch. Regular contributors receive push access to unprotected branches,
which simplifies the contribution process.

## 3. Step-by-step guide

<!-- Here's a step-by-step guide for making contributions via PR, adhering to the estimagic
style guide: -->

1. Fork the [estimagic repository](https://github.com/OpenSourceEconomics/estimagic/).
   This will create a copy of the repository where you have write access.

```{note}
As a regular contributor, **clone** the [repository](https://github.com/OpenSourceEconomics/estimagic/) to your local machine and create a new local branch where you implement your changes. You can push your branch directly to the remote estimagic repository and open a PR from there.
```

2. Clone the (forked) repository to your disc. You'll implement all changes and fixes
   there.

1. Open the terminal and execute the following commands from the root directory of your
   local estimagic repository

   ```console
   conda env create -f environment.yml
   conda activate estimagic
   pre-commit install
   ```

   This automatically installs estimagic in editable mode and enables pre-commit hooks
   for linting and stylistic error-checking.

1. Implement your fix or feature. Use git to add, commit, and push your changes to the
   remote repository. To learn more about git, how to stage and commit your work, have a
   look at these
   [online materials](https://effective-programming-practices.vercel.app/git/staging/objectives_materials.html).

1. Contributions are validated in two ways. First, we employ a comprehensive test suite
   to check if new implementations are compatible with estimagic's existing codebase.
   Second, we use
   [pre-commit hooks](https://effective-programming-practices.vercel.app/git/pre_commits/objectives_materials.html)
   to ensure contributions meet our quality standards and adhere to our stylistic
   guidelines.

   Run the test suit with

   ```bash
   pytest
   ```

   Look at the summary report and fix any errors that pop up.

   With pre-commit installed, linters are executed before each commit. A commit is
   rejected if any of the checks fails. Note that some linters fix the errors
   automatically by modifying the code in-place. Restage the respective files.

```{tip}
Skip the next paragraph if you haven't worked on the documentation.
```

6. Assuming you have updated the documentation, check if it builds correctly. Go to the
   root directory of your local estimagic repo and navigate to the `docs` folder. There,
   you need to create the `estimagic-docs` environment and activate it

   ```console
   conda env create -f rtd_environment.yml
   conda activate estimagic-docs
   ```

   Inside the `docs` folder, type

   ```console
   make html
   ```

   which automatically builds the html documentation. All files are saved in the folder
   `build\html`. You can open the html documentation with your browser of choice (e.g.
   Google Chrome). E.g., use the following path to open the index page

   ```console
   build/html/index.html
   ```

   From the index page, you can navigate to any other section of the estimagic
   documentation. Similarly, you can open specific pages directly, e.g.,

   ```console
   $ build/html/explanations/optimization/why_optimization_is_hard.html
   ```

1. Once all tests and pre-commit hooks pass locally, push your changes to the forked
   estimagic repository, and create a pull request through the GitHub interface: Go to
   the Github repository of your fork. A banner will be displayed asking you whether you
   would like to create a pull request. Click on the link.

   ```{note}
   Regular contributors with push access can push their local branch to the remote estimagic repository directly and start a PR from there.
   ```

   Follow the instructions of the estimagic
   [PR template](https://github.com/OpenSourceEconomics/estimagic/blob/main/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md)
   to describe your contribution, the problem you address, and your proposed solution.

   Opening a PR starts a complete run of the test suite on a
   [Continuous Integration (CI)](https://docs.github.com/en/actions/automating-builds-and-tests/about-continuous-integration)
   server. Along with `pytest`, the CI workflow includes linters, code coverage checks,
   doctests, and builds the html documentation. The status of the CI run is shown on
   your PR page. If any errors pop up, make the corresponding modifications to your code
   and reiterate until all CI tests have passed on the remote machine.

1. Ask one of the main contributors to review your changes. Make sure all CI tests pass
   before you request a review. If the reviewer(s) have any remarks or suggestions,
   address them, and commit your modifications.

1. Once you're PR is approved, one of the main contributors will merge it into
   estimagic's main branch.
