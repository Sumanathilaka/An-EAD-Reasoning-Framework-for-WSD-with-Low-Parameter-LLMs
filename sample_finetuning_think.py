import os
import torch
import torch._dynamo
import re
import logging

# Disable dynamo compilation to avoid attention interface conflicts
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from peft import PeftModel
from huggingface_hub import login

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Login to Hugging Face
login("")

# Read data from file
with open('senses.txt', 'r',encoding="utf8") as file:
    data = file.read()

# Split the data into entries based on empty lines
entries = data.strip().split('\n\n')

# Create a list of lists for each entry's details
list_of_lists = []
for entry in entries:
    details = entry.split('\n')
    entry_list = []
    for detail in details:
        _, value = detail.split(':', 1)
        entry_list.append(value.strip())
    list_of_lists.append(entry_list)

def retrieve_meanings(word, data):
    meanings_dict = {}
    for entry in data:
        if word == entry[0].split(".")[0]:
            if entry[-1] !="":
                meanings_dict[entry[0]] = entry[2]+", synonyms :"+entry[-1]
            else:
                meanings_dict[entry[0]] = entry[2]
    return meanings_dict

#Function to identify the WSD word from the given sentence and return the WSD word on a sentence
def wsdword_finder(text):
    match = re.search(r'<WSD>(.*?)</WSD>', text)
    if match:
        word_inside_wsd = match.group(1)
        return word_inside_wsd

print("Program started")
print("CUDA available:", torch.cuda.is_available())
print("Total GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")

print("Gemma 4b finetune model loading...")

# Model configuration
model_name = "google/gemma-3-4b-it"
adapter_repo = "finetuned-gemma-3-4b-it-WSD_Think"
access_token = ""

# Chat-style prompt template for Gemma 4b
CHAT_PROMPT = """
<start_of_turn>user\n{instruction}    {input} <end_of_turn>\n
<start_of_turn>model   {output}"""

EOS_TOKEN = None  # Will be set after tokenizer loads

# Load tokenizer directly from Hugging Face
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=access_token,
    padding_side="right",
    use_fast=True,
)

tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token

# Load base model directly from Hugging Face
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=access_token,
    device_map="auto",
    torch_dtype="auto",
    load_in_4bit=False
)

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, adapter_repo, token=access_token)
print("Model loaded successfully!")

# Initialize text streamer
text_streamer = TextStreamer(tokenizer)

def generate_text(prompt, word_amb, sentence):
    """Generate text using the same format as training with robust error handling"""
    instruction = "You are a linguistic assistant. Given a sentence with an ambiguous word marked with <WSD> tags. identify its correct meaning using proper reasoning chain of thought process."
    
    # Use the same chat prompt format as training
    full_prompt = CHAT_PROMPT.format(instruction=instruction, input=prompt, output="")
    
    # Tokenize the input with length checking
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=2048)
    
      
            
    # More robust generation parameters
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
        
        # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the model's response (after the last <start_of_turn>model)
    if "<start_of_turn>model" in response:
        assistant_response = response.split("<start_of_turn>model")[-1].strip()
            # Remove any trailing <end_of_turn> tokens
        assistant_response = assistant_response.replace("<end_of_turn>", "").strip()
        return assistant_response
        
    return response


def main():
    print("\nStarting inference...")
    
    try:
        with open("corner_1050.txt", "r", encoding="latin-1") as file:
            for line_num, line in enumerate(file, 1):
                try:
                    # Skip empty lines
                    if not line.strip():
                        continue
                        
                    # Check if line has enough parts
                    parts = line.split("\t")
                    if len(parts) < 2:
                        print(f"Line {line_num}: Insufficient data parts")
                        continue
                    
                    sentence = parts[0]
                    wordw = wsdword_finder(sentence)
                    
                    # Add check for wordw being None
                    if wordw is None:
                        print(f"Line {line_num}: No WSD word found in sentence")
                        continue
                    
                    senseid = parts[1].strip()
                    
                    # Validate senseid format
                    if '.' not in senseid:
                        print(f"Line {line_num}: Invalid senseid format: {senseid}")
                        continue
                        
                    wsdWord = senseid.split(".")[0]
                    pos = senseid.split(".")[1]
                    meanings = retrieve_meanings(wsdWord, list_of_lists)
                    
                    if not meanings:
                        print(f"Line {line_num}: No meanings found for word '{wsdWord}'")
                        continue
                        
                    filtered_definitions = {key: value for key, value in meanings.items() if pos in key}
                    
                    if not filtered_definitions:
                        print(f"Line {line_num}: No definitions found for POS '{pos}'")
                        # Use all meanings as fallback
                        filtered_definitions = meanings

                    prompt = (
                        f"{wordw} is an ambiguous word in the sentence.\n"
                        f"Think step by step to identify the meaning of word {wordw}.\n\n"
                        f"The sentence is: {sentence}\n\n"
                        f"Possible meanings are {filtered_definitions}.\n\n"   
                    )
                    
                    # Generate response using the same format as training
                    full_text = generate_text(prompt, wordw, sentence)
                    
                    structured_response = str(full_text)
                    print(f"Line {line_num} Output:", structured_response.replace('\n', ' '))
                    
                    # Clear GPU cache more frequently to prevent memory issues
                    if torch.cuda.is_available() and line_num % 5 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    logger.error(f"Detailed error for line {line_num}: {str(e)}")
                    continue
                    
    except FileNotFoundError:
        print("Error: corner_1050.txt file not found!")
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    main()