# An Exploration-Analysis-Disambiguation Reasoning Framework for Word Sense Disambiguation with Low-Parameter LLMs

An Exploration–Analysis–Disambiguation (EAD) Framework for Word Sense Disambiguation using <4B Parameter Models.

Paper: An Exploration-Analysis-Disambiguation Reasoning Framework for Word Sense Disambiguation with Low-Parameter LLMs  
Authors: Deshan Sumanathilaka, Nicholas Micallef, Julian Hough  
Affiliation: Swansea University, UK  
Conference: LREC 2026 (Accepted)

---

## Overview

This repository contains the implementation of the Exploration–Analysis–Disambiguation (EAD) reasoning framework for Word Sense Disambiguation (WSD).

We demonstrate that low-parameter LLMs (<4B parameters) can achieve competitive and state-of-the-art performance when equipped with structured reasoning strategies:

- Chain-of-Thought (CoT) reasoning  
- Neighbour-word semantic analysis  
- Correct vs incorrect sense elimination  
- Syntactic evidence for verb disambiguation  

Our best-performing models:

- Gemma 3 4B  
- Qwen 3 4B  

achieve performance comparable to larger GPT-based approaches while using significantly fewer parameters.

---

## Key Contributions

- Introduce the EAD reasoning framework for WSD  
- Release three reasoning-enhanced WSD datasets  
- Demonstrate that structured reasoning is more impactful than increasing parameter size  
- Achieve strong zero-shot generalization  
- Reduce training data requirements (advanced reasoning uses only 10% of CoT data)

---

## EAD Framework

The framework consists of three sequential phases:

### 1. Exploration
Collect candidate senses for the ambiguous word.

### 2. Analysis
Perform structured reasoning using:
- Semantically relevant neighbouring words
- Contextual evidence
- Justification of the correct sense
- Elimination of incorrect senses

### 3. Disambiguation
Return the final sense ID.

---

## Datasets Used

- FEWS – primary training and evaluation dataset  
- SemCor – supplementary training corpus  
- Fool Me If You Can – adversarial robustness benchmark  
- hardEN – unsolved WSD benchmark  
- 42D – rare and domain-specific sense benchmark  

---

## Training Setup

We use:

- Supervised Fine-Tuning (SFT)
- LoRA adapters
- HuggingFace `transformers` and `trl`
- NVIDIA A100 40GB GPU

### Hyperparameters

```yaml
batch_size: 4
gradient_accumulation: 8
learning_rate: 2e-4
epochs: 2
optimizer: AdamW
scheduler: linear
seed: 3407
```
## Citation

If you use this work, please cite:

```bibtex
@inproceedings{sumanathilaka2026ead,
  title     = {An Exploration-Analysis-Disambiguation Reasoning Framework for Word Sense Disambiguation with Low-Parameter LLMs},
  author    = {Sumanathilaka, Deshan and Micallef, Nicholas and Hough, Julian},
  booktitle = {Proceedings of the Thirteenth Language Resources and Evaluation Conference},
  year      = {2026}
}
```

# Acknowledgement 

Supported by:
Supercomputing Wales (ERDF via Welsh Government)
EPSRC Grant EP/X009343/1 (FLUIDITY)
