window.MathJax = {
  // \boldsymbol lives in an optional extension; without this it renders as
  // literal red text instead of a bold vector.
  loader: { load: ["[tex]/boldsymbol"] },
  tex: {
    packages: { "[+]": ["boldsymbol"] },
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true
  },
  options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" }
};
document$.subscribe(() => { MathJax.startup.output.clearCache(); MathJax.typesetClear(); MathJax.texReset(); MathJax.typesetPromise(); });
