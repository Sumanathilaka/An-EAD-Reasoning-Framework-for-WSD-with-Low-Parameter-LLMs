
import os
import torch
import torch._dynamo

# Disable dynamo compilation to avoid attention interface conflicts
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login

# Login to Hugging Face
login("")

print("Program started")
print("CUDA available:", torch.cuda.is_available())
print("Total GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

print("Gemma-3 4B finetune model loading...")

# Model configuration
model_name = "google/gemma-3-4b-it"
adapter_repo = "finetuned-gemma-3-4b-it-WSD"
access_token = ""

# Chat-style prompt template for Gemma 4B (same as training)
CHAT_PROMPT = """
<start_of_turn>user\n{instruction}    {input} <end_of_turn>\n
<start_of_turn>model   {output} <end_of_turn>"""

EOS_TOKEN = None

def extract_word(sentence):
    """Extract word from <WSD> tags and clean sentence - same as training code"""
    start_index = sentence.find('<WSD>')
    end_index = sentence.find('</WSD>')
    if start_index != -1 and end_index != -1:
        word = sentence[start_index + len('<WSD>'):end_index]
        cleaned_sentence = sentence[:start_index] + word + " " + sentence[end_index + len('</WSD>'):]
        return word, cleaned_sentence.strip()
    return None, sentence

def extract_sentence(sentence):
    """Extract cleaned sentence without word - for backward compatibility"""
    word, cleaned_sentence = extract_word(sentence)
    return cleaned_sentence

# Load tokenizer directly from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=access_token,
    padding_side="right",
    use_fast=True,
)

tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=access_token,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=False
)

model = PeftModel.from_pretrained(base_model, adapter_repo, token=access_token)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Important: Set model to evaluation mode
model.eval()
print("Model loaded successfully!")


def generate_text(word_amb, sentence):
    """Generate text using the same format as training"""
    instruction = "Identify the meaning of the word in the sentence."
    input_text = f"Below is the sentence which contain ambiguous word. Identify the meaning of the word based on the context of the sentence. The answer expects the meaning of the ambigous word in few words only. The word is '{word_amb}'. Sentence: {sentence}."
    
    # Use the same chat prompt format as training
    full_prompt = CHAT_PROMPT.format(instruction=instruction, input=input_text, output="")
    
    # Tokenize the input with proper truncation
    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the new generated part (remove the input prompt)
    prompt_length = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
    result = generated_text[prompt_length:].strip()
    
    return result

def main():
    print("\nStarting inference...")
    
    try:
        with open("corner_1050.txt", "r", encoding="latin-1") as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Parse the input file
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                        
                    sentence1 = parts[0]
                    wsd_word = parts[1].strip()
                    word_amb = wsd_word.split(".")[0]
                    
                    # Extract word and clean sentence using training format
                    extracted_word, sentence = extract_word(sentence1)
                    
                    # Use extracted word if available, otherwise use parsed word
                    if extracted_word:
                        word_amb = extracted_word
                    
                    #print(f"\nProcessing line {line_num}:")
                    #print(f"Word: {word_amb}")
                    #print(f"Sentence: {sentence}")
                    
                    # Generate response using the same format as training
                    full_text=generate_text(word_amb, sentence)
                    
                    structured_response = str(full_text)
                    print("Output:", structured_response.replace('\n', ' '))
                    
                    # Clear GPU cache periodically
                    if torch.cuda.is_available() and line_num % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print("Error: corner_1050.txt file not found!")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()