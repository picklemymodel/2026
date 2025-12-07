---
layout: distill
title: How many tokens does it take to say "नमस्ते"? A Dive into Indic Tokenization
description: Tokenizers trained on English-dominant data often produce unusually high token counts for Indic languages. This "tokenizer fertility" increases sequence lengths, raises compute costs, and can hurt downstream performance, even when the underlying model is strong. In this post, we examine how fertility varies across major Indic scripts and how it affects language modeling quality, inference efficiency, and instruction-following behavior.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

mermaid:
  enabled: true
  zoomable: true

authors:
  - name: Anonymous

bibliography: 2026-04-27-indic-tokenization.bib

toc:
  - name: Equations
  - name: Images and Figures
    subsections:
      - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

## Introduction

To be updated
