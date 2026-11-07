// MathJax 3 configuration for the groundfield documentation. MathJax 3.2.2
// itself is vendored in javascripts/vendor/mathjax/ (see mkdocs.yml).
//
// We need to cover two sources of math in the site:
//   * Markdown pages processed by pymdownx.arithmatex (generic mode) which
//     emits \(...\) for inline and \[...\] for display math.
//   * Jupyter notebooks rendered through mkdocs-jupyter, where math cells
//     still use the raw $...$ and $$...$$ delimiters.
//
// Enabling both delimiter sets in MathJax 3 is enough to render everything.
window.MathJax = {
  tex: {
    inlineMath: [
      ["\\(", "\\)"],
      ["$", "$"],
    ],
    displayMath: [
      ["\\[", "\\]"],
      ["$$", "$$"],
    ],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    // Do not skip any container; arithmatex wraps output in <span class="arithmatex">,
    // while notebook markdown lands in generic <div class="jp-RenderedHTMLCommon">
    // without a dedicated class, so we must allow global processing.
    ignoreHtmlClass: "tex2jax_ignore",
    processHtmlClass: "tex2jax_process",
  },
};

// MkDocs Material uses an instant-loading router that swaps the document body
// without a full page reload; document$ emits the first page and every page
// reached by instant navigation. MathJax typesets the first page itself during
// startup, so the first emission is skipped. Later pages are typeset through
// MathJax.startup.promise, which serialises the calls: two concurrent typeset
// runs (startup typeset waiting for an autoloaded extension such as
// \boldsymbol, plus a second typesetPromise) collide in MathJax 3.2 with
// "Cannot read properties of null (reading 'replaceChild')".
if (typeof document$ !== "undefined") {
  let firstDocument = true;
  document$.subscribe(() => {
    if (firstDocument) {
      firstDocument = false;
      return;
    }
    const mathjax = window.MathJax;
    if (!mathjax || !mathjax.startup || !mathjax.startup.promise) {
      return;
    }
    mathjax.startup.promise = mathjax.startup.promise
      .then(() => {
        mathjax.typesetClear();
        mathjax.texReset();
        return mathjax.typesetPromise();
      })
      .catch((error) => console.error("MathJax typesetting failed:", error));
  });
}
