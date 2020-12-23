import nltk
nltk.download('punkt')
import torch
from transformers import BertTokenizer
from MyPredictor import ABSApredictor

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/home/', methods=['GET'])
def return_home():
    return "This is the LexiTech API."

@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    pred_dict = {}
    for i, sent in enumerate(nltk.sent_tokenize(data["text"], "german")):
        prediction = predictor.predict(sent)
        pred_dict["sent{}".format(i)] = {"text": sent, "pred": prediction}
    return jsonify(pred_dict)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=False)

    model = torch.load(r"./model_prod_BERT_trimsweep.pt")
    model.eval()

    predictor = ABSApredictor(model, tokenizer, 493, device)

    app.run(host='0.0.0.0')
