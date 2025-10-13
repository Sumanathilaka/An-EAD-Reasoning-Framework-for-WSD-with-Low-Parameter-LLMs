import os
import torch
import gc
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from trl import SFTTrainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# Model parameters
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def clear_gpu_memory():
    """Comprehensive GPU memory clearing"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


# Initial memory cleanup
print("=== Initial Memory Cleanup ===")
clear_gpu_memory()

# Login to Hugging Face
login("")

print("Program started")
print("CUDA available:", torch.cuda.is_available())
print("Total GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

torch.cuda.set_device(0)  # Set to 1 if you want the second GPU
print(f"Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")


# Model configuration
model_name = "google/gemma-3-4b-it"
max_seq_length = 2048
load_in_4bit = False
access_token = ""

# Configure quantization
if load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
else:
    quantization_config = None

# Load tokenizer directly from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=access_token,
    padding_side="right",
    use_fast=True,
)

tokenizer.pad_token = tokenizer.eos_token

# Load model directly from Hugging Face with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    token=access_token,
    attn_implementation='eager',
)

# Set up LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Apply LoRA to the model
# model = get_peft_model(model, peft_config)

# Chat-style prompt template for Gemma 2B
CHAT_PROMPT = """
<start_of_turn>user\n{instruction}    {input} <end_of_turn>\n
<start_of_turn>model   {output} <end_of_turn>"""

EOS_TOKEN = tokenizer.eos_token

def extract_word(sentence):
    start_index = sentence.find('<WSD>')
    end_index = sentence.find('</WSD>')
    if start_index != -1 and end_index != -1:
        word = sentence[start_index + len('<WSD>'):end_index]
        cleaned_sentence = sentence[:start_index] + word + " " + sentence[end_index + len('</WSD>'):]
        return word, cleaned_sentence.strip()
    return None, sentence

def formatting_prompts_func(examples):
    sentences = examples["input"]
    outputs = examples["output"]
    texts = []
    instruction = "You are a linguistic assistant. Given a sentence with an ambiguous word marked with <WSD> tags ; identify its correct meaning using proper reasoning chain of thought process."
    for sentence, output in zip(sentences, outputs):
        prompt = CHAT_PROMPT.format(instruction=instruction, input=sentence, output=output)
        texts.append(prompt)
    return {"text": texts}

# Load and prepare dataset
dataset = load_dataset("Instruct_Finetune_with_Reasoning_WSD", split="train")
dataset = dataset.map(lambda x: {"input": x["input"], "output": x["output"]})
dataset = dataset.map(formatting_prompts_func, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    num_train_epochs=1,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    report_to="none",
    save_steps=1000,                    
    save_total_limit=2,
)


# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer
)

# Train the model
print("Starting training...")
trainer_stats = trainer.train()
print("Training completed!")


# Save the model
print("\nSaving model...")
model_name_hub = "finetuned-gemma-3-4b-it-WSD_Reason"
model.save_pretrained("./lora_adapters")
tokenizer.save_pretrained("./lora_adapters")
model.push_to_hub(model_name_hub, token=access_token)
tokenizer.push_to_hub(model_name_hub, token=access_token)
print(f"Model successfully saved and pushed to: {model_name_hub}")
print("Training and saving completed successfully!")

