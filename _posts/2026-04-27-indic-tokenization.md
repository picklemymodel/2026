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
  - name: Introduction

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

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <strong>TL;DR.</strong> Some of the important pointers we'll look at, in this post: Are tokenizers failing a billion Indic language speakers? How tokenization bias reinforces linguistic inequality in AI models?
</div>

To be updated
