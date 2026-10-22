(how-to-contribute)=

# How to contribute

## 1. Intro

We welcome your ideas and expertise, whether you are getting started with optimagic or
bring years of experience in optimization and scientific software. There are many ways
to contribute, from suggesting features and reporting bugs to improving documentation,
adding tests, and contributing code.

Our preferred starting point for new features and substantial improvements is a
well-developed issue. It helps us understand the need, agree on a design, and plan the
implementation together.

When a well-developed issue leads to a merged change, we credit the issue author as a
contributor in the corresponding [release notes](changes.md), alongside those who
implement the change.

The situations below explain how to get started and which sections of this guide to
read.

- **You would like a new feature.** Describe your use case and the behavior you would
  find helpful. See [Writing an issue](#contributing-issues).
- **You noticed a bug.** Tell us what happened, what you expected, and how to reproduce
  it. See [Writing an issue](#contributing-issues).
- **You want to improve documentation or tests.** Small, focused improvements are
  welcome as pull requests. Follow [Preparing a change](#contributing-preparing) and
  [Opening and updating a pull request](#contributing-prs). For substantial changes,
  start with an issue to agree on the scope and approach.
- **You want to implement an issue or contribute a code improvement.** Small,
  well-defined changes are welcome as pull requests. For larger changes, first agree on
  the scope and approach with the maintainers on the issue. Then follow
  [Preparing a change](#contributing-preparing) and
  [Opening and updating a pull request](#contributing-prs).

If you use AI tools, read [Using AI tools](#contributing-ai) before posting on GitHub.

If you are looking for a starting point, browse our
[issue tracker](https://github.com/optimagic-dev/optimagic/issues) for tasks that
interest you. You can also work through the documentation and examples and suggest ways
to make them clearer or easier to follow.

(contributing-issues)=

## 2. Writing an issue

Search the issue tracker for related discussions first. An existing issue may already
address your question, and additional examples or evidence can help move it forward.
When opening a new issue, choose the template that best fits your contribution.

### Feature requests

Help us understand the problem you want to solve:

- Describe your use case and why the change would help.
- Explain what you currently do and where that approach falls short.
- Describe the behavior you would like, with a small example if useful.

If you have ideas for an API or implementation, include them so we can discuss the
design together.

### Bug reports

Give us enough information to reproduce and investigate the problem:

- A small, runnable example, or clear steps to reproduce the behavior.
- What you expected and what actually happened, including relevant error messages.
- Your optimagic and Python versions, operating system, and relevant dependency
  versions.

For numerical problems, include the relevant inputs and results so we can assess the
behavior.

### Larger improvements

For substantial changes to code, documentation, or tests, describe the current
limitation, intended outcome, affected area, and proposed scope. Share any approach or
tradeoffs you have considered. Agree on the scope and approach with the maintainers
before starting implementation.

For example, an illustrative documentation issue might say: "In the algorithm-selection
guide, I could not tell how to choose between two derivative-free methods. An example
comparing when each is useful would help." A proposal for reorganizing several guides
could outline the new structure and explain which reader needs it addresses.

If you use AI tools to prepare or submit an issue, consult
[Using AI tools](#contributing-ai) before posting. Once you are ready to implement an
agreed change, continue with [Preparing a change](#contributing-preparing).

(contributing-preparing)=

## 3. Preparing a change

### Set up your working copy

1. Fork the [repository](https://github.com/optimagic-dev/optimagic/), clone your fork,
   and create a branch for your change. Contributors who already have push access can
   use a branch in the main repository.

1. Install [Pixi](https://github.com/prefix-dev/pixi) for Python environments and
   development tasks, and [prek](https://github.com/j178/prek) for pre-commit checks.
   Prek is installed separately from Pixi.

1. From the repository root, enable the Git hooks:

   ```console
   $ prek install
   ```

Pixi automatically selects the environment for the tasks below and creates it when
needed. Use `-e <environment>` to target a particular test environment. Tasks and
environments are defined in `pyproject.toml`. To discover available commands, run:

```console
$ pixi task list
```

### Follow the relevant guides

- For code and tests, follow the [style guide](styleguide.md).
- For typing and public interfaces, consult [EP-02](ep-02-typing.md).
- For documentation, follow the documentation section of the
  [style guide](styleguide.md).

Keep the change focused on its purpose. The current tool configuration is in
`pyproject.toml` and `.pre-commit-config.yaml`.

### Check your change

Run tests for the behavior you changed while iterating. For example:

```console
$ pixi run tests tests/optimagic/optimization/test_optimize.py
```

Include relevant slow tests before opening a PR. For broader checks, use
`pixi run tests-fast`; it excludes slow and JAX tests on every platform. For
cross-cutting changes, run the full suite with `pixi run tests` in an environment with
the required dependencies. The named Linux test environments in `pyproject.toml` include
JAX and PETSc.

For changes affecting typing, run:

```console
$ pixi run ty
```

With hooks enabled, formatting and lint checks run before each commit. Some checks
modify files automatically; review and re-stage those changes before committing again.

### Documentation changes

Build the documentation and inspect the affected pages:

```console
$ pixi run build-docs
```

Open `docs/build/html/index.html` in your browser to view the result. The build executes
notebooks, so allow time for their examples to run.

### Generated code

Changes affecting optimizer definitions, `src/optimagic/algorithms.py`, or `.tools/`
trigger a hook that regenerates `src/optimagic/algorithms.py` from the optimizer
definitions. This hook requires `uv` on your path. Include the regenerated file in your
change.

(contributing-prs)=

## 4. Opening and updating a pull request

Prepare a focused PR that makes your change easy to assess:

- Use a descriptive title, select the appropriate base branch, and follow the
  [PR template](https://github.com/optimagic-dev/optimagic/blob/main/.github/PULL_REQUEST_TEMPLATE/pull_request_template.md).
- Explain the problem, your solution, and any remaining work. Link the relevant issue;
  use `Closes #XXXX` when the PR resolves it.
- Describe the checks you ran, their results, and any remaining validation gaps.
- Identify the author of a well-developed issue that led to the change so they can
  receive contributor credit.

If you use AI tools, read [Using AI tools](#contributing-ai) before publishing or
updating the PR. Include the requested description of substantial AI assistance.

Push your branch to your fork, or to the main repository if you have push access, and
open the PR against optimagic. Automated CI checks cover tests, formatting, typing, and
documentation. Monitor the results, investigate failures, and explain any unresolved
limitations.

Request a maintainer review when the PR is ready. Respond to feedback with changes or an
explanation, and keep the description current as the PR develops. A maintainer will
merge the approved change.

(contributing-ai)=

## 5. Using AI tools

We welcome carefully prepared contributions, including AI-written code and text.
Contributors should check what they submit for accuracy and relevance and be able to
explain it. Review generated code and test the behavior it changes, following
[Preparing a change](#contributing-preparing) and the [style guide](styleguide.md).

We ask contributors to briefly describe substantial AI use in their PR, including how
they used the tools and checked the result. The PR template includes a reminder.

Autonomous agent activity on GitHub requires explicit prior maintainer approval for the
proposed activity. This includes opening issues or PRs, posting comments, and submitting
reviews. Contributors remain responsible for the content and actions of agents they use.

Maintainers prioritize relevant, carefully prepared submissions and may close or remove
submissions at their discretion. Feedback is provided as time allows. Repeated
low-quality or unsolicited activity may lead to restrictions on participation.
