Releases
========

What is the new version number?
-------------------------------

The version number depends on the severity of the changes and adheres to `semantic
versioning <https://semver.org/>`_. The format is x.y.z..

You are also allowed to append ``-rc.1`` after the last digit to indicate the first or
higher release candidates. Thus, you can test deployment on PyPI and release preliminary
versions.


How to release a new version?
-----------------------------

1. At first, we can draft a release on Github. Go to
   https://github.com/OpenSourceEconomics/estimagic/releases and click on "Draft a new
   release". Fill in the new version number as a tag and title. You can write a summary
   for the release, but also do it later. Important: Only save the draft. Do not publish
   yet.

2. Second, create a final PR to prepare everything for the new version. The
   name of the PR and the commit message will be "Release vx.y.z". We need to

   - use ``bumpversion part <dev|patch|minor|major>`` to increment the correct part of
     the version number in all files.
   - update information in ``CHANGES.rst`` to have summary of the changes which
     can also be posted in the Github repository under the tag.

3. Run

   .. code-block:: bash

       $ conda build .

   and check whether you can actually build a new version. If you experience errors, fix
   them here. Depending on whether you allowed automatic upload to Anaconda, the release
   appears under your account. Feel free to delete it.

4. Merge the PR into master.

5. After that, revisit the draft of the release. Make sure everything is fine. Now, you
   click on "Publish release" which creates a version tag on the latest commit of the
   specified branch. Make sure to target the master branch.

6. Check out the tag in your local repository and run

   .. code-block:: bash

       $ conda build . --user OpenSourceEconomics

   In case automatic upload is disabled, copy the path to the built package and type

   .. code-block:: bash

       $ anaconda upload <path> --user OpenSourceEconomics

6. Visit `Anaconda.org <https://anaconda.org/OpenSourceEconomics/estimagic>`_ and check
   whether the release is available.

7. Spread the word!
