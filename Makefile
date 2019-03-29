# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Sphinx Autobuild
livehtml:
	python3 ipynb_to_gallery.py "./blog_content_source/**/*.ipynb" && \
	make html && \
	sphinx-autobuild \
	-b html \
	--open-browser \
	--ignore "blog_content/*" \
	--ignore "blog_content_source/**/*.ipynb" \
	--ignore ".git/**" \
	. "$(BUILDDIR)/html"

clean-cache:
	make clean
	rm -rf blog_content 