# Makefile for Sphinx documentation

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build

# You can set these variables from the command line.
SPHINXPROJ    = Ocean Flow Matching
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html latexpdf linkcheck doctest

help:
	@echo "";
	@echo "Makefile for building the Sphinx documentation";
	@echo "";
	@echo "Targets:";
	@echo "  html       to build the documentation in HTML format";
	@echo "  clean      to remove the build directory";
	@echo "";

clean:
	rm -rf $(BUILDDIR)/*

html:
	KMP_DUPLICATE_LIB_OK=TRUE $(SPHINXBUILD) -b html $(SPHINXOPTS) $(SOURCEDIR) $(BUILDDIR)/html
	@echo "";
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html.";

apidoc:
	sphinx-apidoc -o $(SOURCEDIR)/api ../src
