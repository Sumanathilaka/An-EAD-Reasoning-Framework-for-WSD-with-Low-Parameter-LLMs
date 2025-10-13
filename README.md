# Small-Models-Sharp-Minds-Reasoning-Driven-Word-Sense-Disambiguation-with-Low-Parameter-LLMs


 Word Sense Disambiguation (WSD) remains a key challenge in Natural Language Processing (NLP), especially
 when dealing with rare or domain-specific senses that are often misinterpreted. While modern high-parameter
 Large Language Models (LLMs) such as GPT-4-Turbo have showed state-of-the-art WSD performance, their
 computational and energy demands limit scalability. This study investigates whether low-parameter LLMs (<4B
 parameters) can achieve comperable results through fine-tuning strategies that emphasize reasoning-driven sense
 identification. Using the FEWS dataset augmented with semi-automated, rationale-rich annotations, we fine-tune
 eight small-scale open-source LLMs (e.g., LLama and Qwen). Our results show that CoT-based reasoning combined
 with neighbor-word analysis achieves performance comparable to GPT-4-Turbo in zero-shot settings. Importantly,
 Gemma-3-4B and Qwen-3-4B models consistently outperform all medium-parameter baselines and state-of-the-art
 models on FEWS, with robust generalization to unseen senses. Furthermore, evaluation on the unseen “Fool
 Me If You Can” dataset confirms strong cross-domain adaptability without task-specific fine-tuning. This work
 demonstrates that with carefully crafted reasoning-centric fine-tuning, low-parameter LLMs can deliver accurate WSD
 while substantially reducing computational and energy demands.
