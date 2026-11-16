# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
# -W turns warnings into errors; --keep-going still reports all of them. -j auto reads
# sources (including notebook execution) and writes pages on all available cores.
SPHINXOPTS    = -W --keep-going -j auto
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = optimagic
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
