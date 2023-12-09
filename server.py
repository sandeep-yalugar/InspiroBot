from flask import Flask,request,jsonify
from flask_cors import CORS
import torch
import os
import numpy as np
from torch.nn.functional import cosine_similarity
import openai
from experiments.core.models.huggingface.novelT5 import T5Novel

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import sys
model = T5Novel(model_version='t5-base',num_classes=32,device='cpu')
state_dict = torch.load('./yourcheckpoint/model_checkpoint_new16',map_location='cpu')
model.load_state_dict(state_dict)
model.lm_model.config.dropout_rate = 0
model.to('cpu')

tokenizer = T5Tokenizer.from_pretrained('t5-base')
app = Flask(__name__)
CORS(app)

@app.route('/')
def hello():
    return 'Hello, World!'

def generate(stack):
    # Your logic to process the 'stack' array and generate a response
    # For example, let's just reverse the array for demonstration purposes
    result = stack[::-1]
    return result
def gen_story(prob):
        api_key = "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"
        openai.api_key = api_key

        # Set the prompt to the provided string
        prompt = f'write a philosophical story to console and give a new perspective to a person who explains his problems as {prob} complete the story within 600 tokens.'
        conversation=[
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": prompt}]

        # Generate the response using the ChatGPT API
        response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=conversation,
        temperature=0.5,
        max_tokens=800 # Adjust the token limit based on your needs
        )

        # Extract the story from the response
        story = response.choices[0].message.content

        return story
def generate_reply(stack):
    paragraph_inp = ''.join(stack)
    inputs = tokenizer.encode(paragraph_inp, return_tensors='pt', padding=True)
    inputs = inputs.to('cpu')

    with torch.no_grad():
        outputs = model.lm_model.generate(inputs, max_length=500,temperature=0.7,num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

@app.route('/createReply', methods=['POST'])
def create_reply():
    try:
        data = request.get_json()
        stack = data['stack']  # Assuming 'stack' is a key in the JSON request body

        result = generate_reply(stack)

        # Returning the result in JSON format
        return jsonify({'result': result})
    except Exception as e:
        print("Error occurred while generating reply")
        print(e)
        return jsonify({'result': 'sorry i am unable to reply due to some server error'})
@app.route('/createStory',methods=['POST'])
def create_story():
    try:
        data = request.get_json()
        context = data['context']
        prob = ''
        for i in context:
            prob = prob + i +'.'
        story = gen_story(prob)
        return jsonify({'result': story})
    

    except Exception as e:
        return jsonify({'result': 'sorry i am unable to get u a story because of server error'})

if __name__ == '__main__':
    try:
        app.run(debug=True, port=3005)
        print('successfully running server')
        
    except Exception  as e:
         print(e)
