import nltk

nltk.download('punkt')
import torch
from transformers import BertTokenizer
from MyPredictor import ABSApredictor

from fastapi import FastAPI
from pydantic import BaseModel
import json


class Input(BaseModel):
    text: str

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=False)

model = torch.load(r"./model_prod_BERT_trimsweep.pt")
model.eval()

predictor = ABSApredictor(model, tokenizer, 493, device)

app = FastAPI()


@app.get('/home/')
@app.get("/")
def return_home():
    return "Space for the Lexitech API Documentation."


@app.post('/api/')
async def makecalc(input: Input):
    pred_dict = {}
    for i, sent in enumerate(nltk.sent_tokenize(input.text, "german")):
        prediction = predictor.predict(sent)
        pred_dict["sent{}".format(i)] = {"text": sent, "pred": prediction}
    return pred_dict


# run with: uvicorn main:app --reload