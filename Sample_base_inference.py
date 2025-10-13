import torch
from transformers import pipeline
from huggingface_hub import login

# Login to Hugging Face
login(token="")
print("Gemma 4b model loaded")

import re
def extract_sentence(sentence):
    # Find the start and end index of the <WSD> tags
    start_index = sentence.find('<WSD>')
    end_index = sentence.find('</WSD>')
    if start_index != -1 and end_index != -1:
        # Extract the word between <WSD> tags
        word = sentence[start_index + len('<WSD>'):end_index]
        # Remove <WSD> and </WSD> tags from the sentence
        cleaned_sentence = sentence[:start_index] + word+" " + sentence[end_index + len('</WSD>'):]
        #finding the index of the word
        list_word=cleaned_sentence.split(" ")
        return cleaned_sentence.strip()

pipe = pipeline(
    "text-generation",
    model="google/gemma-3-4b-it",
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",  # replace with "mps" to run on a Mac device
)

def generate_text(prompt):
    
    messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "You are a helpful assistant to identify the meaning of a word in a sentence."}],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    }
]
    user_message = messages[1]["content"][0]["text"]
    output = pipe(text_inputs=user_message, max_new_tokens=256)
    return(output[0]["generated_text"])

def main():
    with open("corner_1050.txt", "r", encoding="latin-1") as file:
        for line in file:
            wsd_word = line.split("\t")[1].strip("\n")
            word_amb = wsd_word.split(".")[0]           
            sentence1 = line.split("\t")[0]
            sentence =extract_sentence(sentence1)

            prompt = (
        f"Identify the meaning of the word '{word_amb}' in the below sentence.\n"
        f"Sentence:{sentence}\n"
	f"Return only the meaning of the word. Do not add extra information."
	)

            full_text=generate_text(prompt)
            #print(prompt)
            
            structured_response = str(full_text)
            print("\noutput:", structured_response.replace('\n', ' '))
            

if __name__ == "__main__":
    main()