// MathJax v3 config for MkDocs Material + Arithmatex, with AMS + both display delimiters
window.MathJax = {
  loader: {
    load: ['[tex]/ams', '[tex]/boldsymbol', '[tex]/textmacros']
  },
  tex: {
    packages: {'[+]': ['ams', 'boldsymbol', 'textmacros']},
    inlineMath: [['\\(', '\\)']],                  // inline: \( ... \)
    displayMath: [['\\[', '\\]'], ['$$', '$$']],   // display: \[...\] or $$...$$
    processEscapes: true,
    processEnvironments: true
  }
};

// Re-typeset on SPA page changes (MkDocs Material)
if (window.document$) {
  document$.subscribe(() => MathJax.typesetPromise());
}
