import torch
import os
import numpy as np
from torch.nn.functional import cosine_similarity
from .expreiments.core.models.huggingface.novelT5.py import T5Novel

from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
import sys


'''def calc_similarity(transcripts, model):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model.eval()

    with torch.no_grad():
        sentence_embeddings = model.encode(transcripts)
        mydict = dict(zip(transcripts, sentence_embeddings))

        cos_sim = []
        for i in range(0, len(transcripts), 3):
            tensor1 = torch.tensor(mydict[transcripts[i+1]]).unsqueeze(0)
            tensor2 = torch.tensor(mydict[transcripts[i+2]]).unsqueeze(0)
            cos_sim.append(cosine_similarity(tensor1, tensor2))

        print("Cosine Similarity: {}".format(np.mean(cos_sim)))

def generate_response(model, tokenizer, device):
    model.eval()

    transcripts = []

    while True:
        inp = input("You: ")
        if inp.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        inputs = tokenizer.encode(inp, return_tensors='pt', padding=True)
        inputs = inputs.to(device)

        with torch.no_grad():
            outputs = model.generate(inputs, max_length=50, num_return_sequences=1)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Chatbot:", response)
        sys.stdout.flush()

        transcripts.extend([inp, response])

    return transcripts'''

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(DEVICE)

    model = T5Novel(model_version='t5-base',
                            num_classes=32,
                            device=DEVICE)
    state_dict = torch.load('./yourcheckpoint/model_checkpoint_Best',map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(DEVICE)

    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    while True:
        model.eval()
        inp = input("You: ")
        if inp.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        inputs = tokenizer.encode(inp, return_tensors='pt', padding=True)
        inputs = inputs.to(DEVICE)

        with torch.no_grad():
            outputs = model.lm_model.generate(inputs, max_length=50, num_return_sequences=1)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Chatbot:", response)
        sys.stdout.flush()

if __name__ == "__main__":
    main()
