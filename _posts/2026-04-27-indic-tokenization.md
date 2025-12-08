---
layout: distill
title: How many tokens does it take to say “नमस्ते”? A Dive into Indic Tokenization
description: Tokenizers trained on English-dominant data often produce unusually high token counts for Indic languages. This 'tokenizer fertility' increases sequence lengths, raises compute costs, and can hurt downstream performance, even when the underlying model is strong. In this post, we examine how fertility varies across major Indic scripts and how it affects language modeling quality, inference efficiency, and instruction-following behavior.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous
    url: ""
    affiliations:
      name: Anonymous

bibliography: 2026-04-27-indic-tokenization.bib

toc:
  - name: What're tokens? and why do they matter so much to us?
    subsections:
      - name: Introduction
      - name: Tokenizer Fertility - A Key Measure of Tokenization Efficiency
  - name: The Indic Problem!
    subsections:
      - name: How high fertility tokenizers make IndicNLP unfair?
  - name: Quantifying the Fertility Tax: Evidence from Indic-GEC
    subsections:
      - name: The Fertility Gap
      - name: The Downstream Impact
  - name: Conclusion

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
  <strong>TL;DR.</strong> Are tokenizers failing a billion Indic language speakers? How tokenization bias reinforces linguistic inequality in AI models? In this post, we discuss the hidden bottlenecks in Multilingual AI.
  <div style="margin-top: 10px;">
    <span style="background-color: #e0e0e0; color: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-right: 5px;">#tokenizer-fertility</span>
    <span style="background-color: #e0e0e0; color: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; margin-right: 5px;">#LLMs</span>
    <span style="background-color: #e0e0e0; color: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">#IndicNLP</span>
    <span style="background-color: #e0e0e0; color: #333; padding: 2px 8px; border-radius: 4px; font-size: 0.85em;">#low-resource-languages</span>
  </div>
</div>

## What're tokens? and why do they matter so much to us?

### Introduction

In recent years, the field of Natural Language Processing (NLP) has been revolutionized by advances in Large Language Models (LLMs). <d-cite key="hagos2024recentadvancesgenerativeai"></d-cite> Trained on large text corpora, these AI models can understand, generate and manipulate language in a human-like way, enabling a wide range of tasks without task-specific supervision. Simply, an LLM takes an input (prompt) and generates the output. They’re like magic, no?

But how do they operate? How do they see strings? LLMs do not operate directly on raw text. Instead, when a text is fed into the model, before interpreting, it’s first segmented into a series of multi-letter chunks called `tokens`. The process of converting text into these units is called tokenization.

### Tokenizer Fertility - A Key Measure of Tokenization Efficiency

Most modern LLMs use subword tokenizers. These tokenizers learn a vocabulary of common text fragments from a large corpus. At inference time, each word is broken into the longest possible fragments from this vocabulary. For languages with predictable morphology or large amounts of training data (like English), this strategy works reasonably well. However, tokenization performance varies across languages. Now, if a tokenizer was mainly trained on high-resource languages, it might've failed to learn useful subword units for languages that were underrepresented in its training data. When this happens, a single word may split into many tokens. This effect is called high fertility.

**Tokenizer fertility** $(F)$ (a measure of how many tokens a model generates per source word) can be defined by:

$$\text{F}(L) = \frac{1}{|D_L|} \sum_{s \in D_L} \frac{\text{Count}_{\text{tokens}}(s)}{\text{Count}_{\text{words}}(s)}$$

where $|D_L|$ refers to the number of sentences in a dataset $D$ of language $L$ and $s$ is a sentence in the dataset.

Now, More tokens $\leadsto$ longer sequences, $\implies$ less context fits into the model's fixed window. Fragmented words give the model fewer stable patterns to learn, reducing sample efficiency.

Simply, you can consider tokens as the "units of thought" the model works with. If those units are poorly aligned with a language, the model begins at a disadvantage -- before any actual modeling even starts!

## The Indic Problem!

{% include figure.liquid path="assets/img/2026-04-27-indic-tokenization/banner.jpg" class="img-fluid" %}
<div class="caption">
    Credits: Generated by Gemini Nano Banana
</div>

[Indic languages](https://en.wikipedia.org/wiki/Indo-Aryan_languages) represent **one of the world's largest and most diverse linguistic families**, spoken by hundreds of millions across South Asia. The diversity of these languages reflects the philosophy of Vasudhaiva Kutumbakam – "the world is one family". Indic languages form a critical yet underserved segment of the NLP landscape. This reminds us that technology should serve all languages, not just the high-resource ones. Building language models that handle Indic languages effectively is a step toward more inclusive AI, ensuring that speakers of every language can benefit from advances in LLMs.

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

---

Let's try some mini experiments.

How Fertile Is Your Tokenizer? Visualising the splits:

{% highlight python %}
from transformers import AutoTokenizer
import pandas as pd

def visualize_splits(text, model_names):
    results = []
    for name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(name)
        tokens = tokenizer.tokenize(text)
        
        split_view = " \ ".join([t.replace('Ġ', '').replace(' ', '') for t in tokens])
        results.append({
            "Model": name.split("/")[-1],
            "Token Count": len(tokens),
            "Split View": split_view
        })
    return pd.DataFrame(results)

text = "संप्रभुता" 

models = [
    "microsoft/Phi-3-mini-4k-instruct",
    "google/gemma-2-2b", 
    "ai4bharat/indic-bert"
]

df = visualize_splits(text, models)
print(df.to_markdown(index=False))
{% endhighlight %}

To understand the source of high fertility, we tokenized the Hindi word for "Sovereignty" (संप्रभुता). This word serves as a stress test due to its use of conjuncts (yuktakshars) and vowel modifiers (matras).

| Model                  |   Token Count | Split View                        |
|:-----------------------|--------------:|:----------------------------------|
| Phi-3-mini-4k-instruct |            10 | ▁ \ स \ ं \ प \ ् \ र \ भ \ ु \ त \ ा |
| gemma-2-2b             |             4 | सं \ प्र \ भु \ ता                    |
| indic-bert             |             3 | ▁सप \ रभ \ त                      |

The difference in segmentation strategies is distinct:

**Phi-3-mini (10 Tokens)**: _Orthographic Decomposition_ - The tokenizer fails to recognize Indic subwords, reverting to character-level segmentation. Notably, it splits the conjunct 'प्र' (pra) into three constituent parts: the consonant प, the halant ्, and the consonant r र. The model essentially processes the text as a stream of unicode distincts rather than linguistic units.

Takeaway: When we say English-centric models are "inefficient" for Indic languages, we aren't just talking about higher API costs, we are talking about semantic dilution. When a single concept like "sovereignty" is stretched across 10 tokens, the relationship between the subject and the verb (which might be 20 words away) becomes mathematically more distant, making complex reasoning significantly harder.

**Gemma-2 (4 Tokens)**: Syllabic Preservation - The tokenizer aligns with the structure of the script, preserving full syllables ("Aksharas"). The complex clusters प्र (pra) and भु (bhu) are treated as single tokens.

Coming to AI4Bharat's IndicBERT, the result might seem great at first glance. However, if you look closely at the split view: सप (Sap), रभ (Rabh), त (Ta), you'll notice that the vowels have disappeared. The tokenizer has achieved this low fertility by performing aggressive normalization.

The Trade-off: The model is extremely efficient, but potentially loses critical semantic information (tense, gender, and root meaning) stored in the vowels. This serves as a crucial lesson: Low fertility is only a virtue if it preserves information.

---

Does high fertility dilute the model's grasp of context?

{% highlight python %}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import pandas as pd
import gc

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

def measure_fertility_perplexity(text, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    token_count = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs.input_ids)
        perplexity = torch.exp(outputs.loss).item()

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "Model": model_id.split("/")[-1],
        "Token Count": token_count,
        "Perplexity": round(perplexity, 2)
    }

#eng_text = "Artificial intelligence is transforming the world."
hindi_text = "कृत्रिम बुद्धिमत्ता दुनिया को बदल रही है।"

models = ["microsoft/Phi-3-mini-4k-instruct", "google/gemma-2-2b"]

results = [measure_fertility_perplexity(hindi_text, m) for m in models]

print(pd.DataFrame(results).to_markdown(index=False))
{% endhighlight %}

We hypothesized that high fertility would confuse the model, leading to higher perplexity (uncertainty). To test this, we compared [Microsoft Phi-3](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (English-centric tokenizer) against [Google Gemma-2](https://huggingface.co/google/gemma-2-2b) (Multilingual tokenizer) on a Hindi sample. The results, at first glance, seem to defy logic:

| Model                  |   Token Count |   Raw Perplexity |
|:-----------------------|--------------:|-----------------:|
| Phi-3-mini-4k-instruct |            44 |             5.65 |
| gemma-2-2b             |            14 |            44.73 |

Does this mean the English-centric Phi-3 is 8x "smarter" at Hindi than the multilingual Gemma-2? Absolutely not. This is a classic example of the Tokenization Bias in evaluation metrics.

Perplexity measures the average uncertainty per token.

- Phi-3: Because it fragments the Hindi sentence into 44 tiny bytes, many of its prediction steps are trivial. For example, once it predicts the first byte of a character, the subsequent bytes are deterministic. These "easy wins" lower the average perplexity, masking the fact that the model may not grasp the sentence's semantic meaning.

- Gemma-2: With a richer vocabulary, Gemma represents the sentence in just 14 dense tokens. Each prediction requires choosing the correct word or root from a large set, so each step is harder. 

To compare them fairly, we normalize perplexity by word count, not token count.

$$\text{PPL}_{\text{word}} = \text{PPL}_{\text{token}}^{(\text{Token Count} / \text{Word Count})}$$

Thus, Phi-3 Normalized: $5.65^{(44/7)} \approx 5.65^{6.28} \approx \mathbf{52,800}$

Gemma-2 Normalized: $44.73^{(14/7)} \approx 44.73^{2.0} \approx \mathbf{2,000}$

The Reality: When normalized, Gemma-2 is actually orders of magnitude better at predicting the sequence than Phi-3.

---

## Quantifying the Fertility Tax: Evidence from Indic-GEC

To move beyond theoretical efficiency, we evaluated tokenization fertility on a multi-lingual dataset <d-cite key="bhattacharyya-bhattacharya-2025-leveraging"></d-cite> designed for Grammatical Error Correction (GEC). The dataset covers five major Indic languages across two script families:

Indo-Aryan: Hindi (`HI`) - Devanagari, Bengali (`BN`) - Eastern Nagari

Dravidian: Tamil (`TAM`), Malayalam (`MAL`) and Telugu (`TEL`)

The dataset serves as a rigorous stress test, featuring inputs laden with compound errors ranging from surface-level spelling and punctuation to intricate morphological challenges like verb conjugation, tense, aspect, agreement with the subject and gender agreement. Each sample often necessitates multiple, interdependent corrections, requiring the model to resolve deep syntactic inconsistencies while preserving semantic intent.

We employed three SoTA LLMs for inference-only GEC using role-based instruction-following prompts: [GPT-4.1 Mini](https://platform.openai.com/docs/models/gpt-4.1-mini), Gemini-2.5-Flash <d-cite key="comanici2025gemini25pushingfrontier"></d-cite>, and [Llama-4-Maverick-17B](https://ai.meta.com/blog/llama-4-multimodal-intelligence/).

### The Fertility Gap

We calculated the fertility rate ($F_{rel}$) for each model across the test set. The results reveal a significant disparity in how these architectures handle Indic scripts.

{% include figure.liquid path="assets/img/2026-04-27-indic-tokenization/comparison.jpg" class="img-fluid" %}

**Observation**: While all models show increased fertility for Dravidian languages, Llama-4-Maverick exhibits extreme fragmentation. A fertility score of <span style="color:red;">5.88</span> for Tamil implies the model needs to process roughly 2.3X more tokens than Gemini 2.5 Flash to correct a Tamil sentence. This indicates that, despite Llama 4’s strong reasoning abilities, its tokenizer slows down inference, raises costs for Dravidian GEC tasks—likely because it falls back to byte-level encoding more often for these scripts. The inflation not only increases latency but also drives the effective inference cost up by a similar multiplier, making Tamil processing disproportionately more expensive on Llama-4-Maverick compared with more efficient models. In contrast, Gemini-2.5-Flash remains significantly more efficient (2.54), likely due to a vocabulary better optimized for non-Latin scripts.

### The Downstream Impact

Does this "tax" matter for quality? <d-cite key="ali-etal-2024-tokenizer"></d-cite> demonstrated that applying English-centric tokenizers to multilingual models can inflate training costs by up to $68\%$ while severely degrading downstream performance. We tested this by measuring the models' ability to correct grammatical errors, maximizing the log-probability of the corrected sequence $y$ given the input $x$.

$$\mathcal{L}_{\text{GEC}}(\theta) = - \sum_{i=1}^{N} \sum_{t=1}^{|y^{(i)}|} \log P_{\theta} ( y_t^{(i)} \mid y_{<t}^{(i)}, x^{(i)} )$$

Evaluation on the test set (using $F_0.5$ and BERT-Score) shows a **clear inverse correlation** between fertility and performance!

{% include figure.liquid path="assets/img/2026-04-27-indic-tokenization/comparison2.jpg" class="img-fluid" %}

<div class="caption">
   Performance of different models on the test set across languages. (PS. The plot has been generated by NanoBanana.)
</div>


Gemini-2.5-Flash, that  uses a `SentencePiece` based multimodal tokenizer, consistently achieved the lowest fertility and high GEC scores across all five languages. Llama-4-Maverick, despite being a capable model, its performance degraded on Tamil and Malayalam -- the exact languages where its tokenization was most inefficient ($>4.5$).

These results suggest that _tokenization density is a bottleneck_. When a model is forced to predict 6 tokens to express one word, the effective context window shrinks and the attention mechanism struggles to model long-range dependencies required for grammatical correction. High fertility is not just a cost issue, it is a quality ceiling.


## Conclusion

To wind up, our analysis highlights that tokenization is not merely a preprocessing step but a structural bottleneck for Indic LLMs. Experiments on the GEC task reveal that model scale cannot fully compensate for poor representation. Even powerful architectures falter when their input is fragmented, while more efficient tokenizers correlate with superior downstream performance.

Moving forward, research must pivot to granular analysis—identifying exactly which words suffer from over-fragmentation and how this breakage distorts sentence-level context. We need to understand the specific failure modes of current vocabularies to fix deeper biases. An additional dimension that needs deeper investigation is the sensitivity of LLMs to sentence-level context when operating under tokenizers with differing fertility. For the research community, the takeaway is clear: we cannot simply scale our way out of inefficient tokenization. True multilingual competence requires optimizing the atomic units of our models. To build LLMs that serve global languages equitably, we must move beyond English-dominant vocabularies and design tokenizers that respect the distinct morphological structures of other low-resource language scripts.