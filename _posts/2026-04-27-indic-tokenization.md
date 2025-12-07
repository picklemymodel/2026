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
  - name: What're tokens? and why do they matter so much to us?
  - name: The Indic Problem!
    subsections:
      - name: How high fertility tokenizers make IndicNLP unfair?
  - name: 

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

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <strong>TL;DR.</strong> Some of the important pointers we'll look at, in this post: Are tokenizers failing a billion Indic language speakers? How tokenization bias reinforces linguistic inequality in AI models?
</div>

## What're tokens? and why do they matter so much to us?

In recent years, the field of Natural Language Processing (NLP) has been revolutionized by advances in Large Language Models (LLMs). <d-cite key="hagos2024recentadvancesgenerativeai"></d-cite> Trained on large text corpora, these AI models can understand, generate and manipulate language in a human-like way, enabling a wide range of tasks without task-specific supervision. Simply, an LLM takes an input (prompt) and generates the output. They’re like magic, no?

But how do they operate? How do they see strings? LLMs do not operate directly on raw text. Instead, when a text is fed into the model, before interpreting, it’s first segmented into a series of multi-letter chunks called `tokens`. The process of converting text into these units is called tokenization.

Most modern LLMs use subword tokenizers. These tokenizers learn a vocabulary of common text fragments from a large corpus. At inference time, each word is broken into the longest possible fragments from this vocabulary. For languages with predictable morphology or large amounts of training data (like English), this strategy works reasonably well. However, tokenization performance varies across languages. Now, if a tokenizer was mainly trained on high-resource languages, it might've failed to learn useful subword units for languages that were underrepresented in its training data. When this happens, a single word may split into many tokens. This effect is called high fertility.

**Tokenizer fertility** $(F)$ (a measure of how many tokens a model generates per source word) can be defined by:

$$\text{F}(L) = \frac{1}{|D_L|} \sum_{s \in D_L} \frac{\text{Count}_{\text{tokens}}(s)}{\text{Count}_{\text{words}}(s)}$$

_where ${|D_L|}$ refers to the number of sentences in a dataset $D$ of language $L$ and $s$ is a sentence in the dataset._

Now, More tokens $\leadsto$ longer sequences, $\implies$ less context fits into the model's fixed window. Fragmented words give the model fewer stable patterns to learn, reducing sample efficiency.

Simply, you can consider tokens as the "units of thought" the model works with. If those units are poorly aligned with a language, the model begins at a disadvantage -- before any actual modeling even starts!

## The Indic Problem!

[Indic languages](https://en.wikipedia.org/wiki/Indo-Aryan_languages) represent one of the world's largest and most diverse linguistic families, spoken by hundreds of millions across South Asia. The diversity of these languages reflects the philosophy of Vasudhaiva Kutumbakam – "the world is one family". Indic languages form a critical yet underserved segment of the NLP landscape. This reminds us that technology should serve all languages, not just the high-resource ones. Building language models that handle Indic languages effectively is a step toward more inclusive AI, ensuring that speakers of every language can benefit from advances in LLMs.

Training Data Dominance: Tokenization algorithms, such as Byte Pair Encoding (BPE), are primarily trained on massive text corpora dominated by high-resource languages, especially English. The resulting vocabularies are optimized for the structure and script of these dominant languages.

### How high fertility tokenizers make IndicNLP unfair?

<div style="border: 2px solid #ee6b6e; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  Consider "नमस्ते" (trans. namaste), meaning "Hello". When tokenized, it should be preserved as one meaningful token. But what if it was wrongly tokenized as ["Hell", "o"]? The semantic meaning is gone! This is the default reality for Indic languages :<
</div>

- Inefficient Segmentation: When a tokenizer breaks a single character into 3-4 tokens, the model loses the semantic unity of that character. It forces the model to learn character composition rather than word meaning.

Indic scripts overwhelm subword models! For non-Latin-script and morphologically complex or agglutinative languages where words are formed by joining many morphemes (for eg. Dravidian language families), the models struggle to create meaningful tokens as their word forms change frequently and the tokenizer does not capture these variations well. These languages often require significantly more tokens to represent the same semantic content as English. For example, some languages may require up to seven times more tokens per sentence than English. We’ll look into this soon.

- Computational Burden: The token inflation leads to higher computational costs and slower processing times. This creates a systematic disadvantage and an accessibility barrier for speakers of these Indic languages. In commercial AI services that use token-based pricing, Indic languages face disproportionately higher costs for the same task, creating economic barriers to accessing AI technology.

- Reduced Context Utilization: Higher token density means less effective use of a model's fixed context window, which can impair performance on tasks requiring extensive contextual understanding.

Let's look at an example using [OpenAI Tokenizer Playground](https://platform.openai.com/tokenizer). "Day by day, it seems nothing changes, yet soon, everything is different.". A translation of this sentence in Bangla (keeping the unicode codepoint count same) would be "দিনে দিনে মনে হয় কিছুই বদলায় না, কিন্তু খুব শিগগিরই সবকিছুই বদলে যায়।".

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-indic-tokenization/gpt-en.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-indic-tokenization/gpt-bn.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Tokenization in GPT-4o which uses a form of Byte Pair Encoding (BPE) via OpenAI's `tiktoken` library, specifically a highly optimized, large-vocabulary version (`o200k_base`).
</div>

Look at the difference in tokenization. Now, this is for just one sentence (again, this worsens for more complex Languages like Malayalam), imagine a huge dataset with thousands of rows to process for inference – these small differences quickly add up, increasing sequence lengths, affecting model memory and may ultimately impact performance.

## The Fertility "Tax"

## The Downstream Impact

