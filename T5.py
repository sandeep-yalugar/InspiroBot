import torch
import os
import numpy as np
from torch.nn.functional import cosine_similarity
import openai
from experiments.core.models.huggingface.novelT5 import T5Novel

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import sys

openai.api_key = os.environ.get('CHATGPT_API_KEY') #enter your chatGPT api key in envirornment variables



def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)
    def gen_story(prob):
        api_key = os.environ.get('CHATGPT_API_KEY')
        openai.api_key = api_key

        # Set the prompt to the provided string
        prompt = f'write a philosophical story to console and give a new perspective to a person who explains his problems as {prob}'
        conversation=[
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": prompt}]

        # Generate the response using the ChatGPT API
        response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0.5,
        max_tokens=200 # Adjust the token limit based on your needs
        )

        # Extract the story from the response
        story = response.choices[0].message.content

        return story

    model = T5Novel(model_version='t5-base',
                            num_classes=32,
                            device=DEVICE)
    state_dict = torch.load('./yourcheckpoint/model_checkpoint_new18',map_location='cpu')
    model.load_state_dict(state_dict)
    model.lm_model.config.dropout_rate = 0
    #model.eval()
    model.to(DEVICE)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    hist = []
    prob_hist = []
    print("enter the conversation length")
    convlen = int(input())
    flag = 0
    while True:
        
        inp = input("You: ")
        if inp.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        #print(len(hist))
        print(prob_hist)
        if len(prob_hist)>convlen and flag==0:
           flag = 1
           hist = []
           prob = "".join(hist)
           story_ = gen_story(prob)
           print("here is a story that might ease you mind")
           print(story_)
        if len(hist)>6:
            hist = hist[4:]
        inp = inp + '.'
        hist.append(inp)
        prob_hist.append(inp)
        paragraph_inp = "".join(hist)
        inputs = tokenizer.encode(paragraph_inp, return_tensors='pt', padding=True)
        inputs = inputs.to(DEVICE)

        with torch.no_grad():
            outputs = model.lm_model.generate(inputs, max_length=500,temperature=0.7,num_return_sequences=1)
        #print(outputs) 
        #with torch.no_grad():
           #outputs = model(input_ids=inputs,emolabel=[1])
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        hist.append(response)
        #print(outputs)
        print("Chatbot:", response)
        sys.stdout.flush()

if __name__ == "__main__":
    main()
