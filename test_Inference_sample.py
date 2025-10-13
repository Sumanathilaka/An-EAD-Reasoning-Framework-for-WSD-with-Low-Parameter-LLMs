import os
import torch
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

# Login to Hugging Face
login("")

print("Program started")
print("CUDA available:", torch.cuda.is_available())
print("Total GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

torch.cuda.set_device(0)  # Set to 1 if you want the second GPU
print(f"Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Model parameters
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
model = get_peft_model(model, peft_config)

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
    instruction = "Identify the meaning of the word in the sentence."
    for sentence, output in zip(sentences, outputs):
        word_amb, cleaned_sentence = extract_word(sentence)
        if word_amb is None:
            word_amb = ""
        # Compose the input for the prompt
        input_text = f"Below is the sentence which contain ambiguous word. Identify the meaning of the word based on the context of the sentence.The word is '{word_amb}'. Sentence: {cleaned_sentence}"
        prompt = CHAT_PROMPT.format(instruction=instruction, input=input_text, output=output)
        texts.append(prompt)
    return {"text": texts}

# Load and prepare dataset
dataset = load_dataset("WSD_DATASET_FEWS", split="train")
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



# Resume from checkpoint if exists
resume_checkpoint_path = "outputs/checkpoint-3170"

# Initialize the SFTTrainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
    tokenizer=tokenizer
)

# Train the model from checkpoint
print(f"Resuming training from checkpoint: {resume_checkpoint_path}")
trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint_path)
print("Training resumed and completed!")


# Save the model
print("\nSaving model...")
model_name_hub = "finetuned-gemma-3-4b-it-WSD"
model.save_pretrained("./lora_adapters")
tokenizer.save_pretrained("./lora_adapters")
model.push_to_hub(model_name_hub, token=access_token)
tokenizer.push_to_hub(model_name_hub, token=access_token)
print(f"Model successfully saved and pushed to: {model_name_hub}")
print("Training and saving completed successfully!")



# Test the trained model with chat-style prompt
print("\nTesting the trained model...")
test_sentence = "I went to <WSD>bank</WSD> to deposit money"
test_word, test_cleaned = extract_word(test_sentence)
test_instruction = "Identify the meaning of the word in the sentence."
test_input = f"The word is '{test_word}'. Sentence: {test_cleaned}"
test_prompt = CHAT_PROMPT.format(instruction=test_instruction, input=test_input, output="")
input_text = test_prompt
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
print("\nModel response:")
outputs = model.generate(
    input_ids=inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=128,
    streamer=text_streamer,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

