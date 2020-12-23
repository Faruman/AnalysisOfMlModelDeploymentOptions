from abc import ABC
import json
import logging
import os

import torch
import torch.nn.functional as F

from transformers import BertForSequenceClassification, BertTokenizer

import nltk
nltk.download('punkt')
import numpy as np
from scipy.special import softmax

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        if tokens_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        tokens = []
        tokens.append("[CLS]")
        for token in tokens_a:
            tokens.append(token)
        tokens.append("[SEP]")

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
            tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask)
        )
    return features

class TransformersClassifierHandler(BaseHandler, ABC):
    """
    Transformers text classifier handler class. This handler takes a text (string) and
    as input and returns the classification text based on the serialized transformers checkpoint.
    """
    def __init__(self):
        super(TransformersClassifierHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        properties = ctx.system_properties
        model_dir = properties.get("model_dir")
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"

        # Read model serialize/pt file
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased', do_lower_case=False)

        self.max_seq_length = 493
        self.model.to(self.device)
        self.model.eval()

        logger.debug('BERT model from path {0} loaded successfully'.format(model_dir))

        self.objective_dict = {"service": "des Services", "style": "des Stils", "food": "des Essens",
                          "quality": "der Qualität", "price": "des Preis", "restaurant": "des Restaurants",
                          "ambience": "des Ambientes", "drinks": "der Drinks", "location": "der Umgebung"}
        self.label_dict = {"positiv": "positive", "neutral": "neutral", "negativ": "negative", "uneinig": "conflict",
                      "unbestimmt": "none"}

        self.initialized = True


    def preprocess(self, data):
        """ Very basic preprocessing code - only tokenizes.
            Extend with your own preprocessing steps as needed.
        """
        text = data[0]["body"]["text"]
        logger.info("Received text: '%s'", text)

        self.sentences = nltk.sent_tokenize(text, "german")

        all_input_ids = []
        all_input_mask = []
        all_features_index = []

        for sentence in self.sentences:
            features = []
            features_index = {}

            for objective in self.objective_dict.keys():
                examples = []
                for label in self.label_dict.keys():
                    text_b = "Die Bewertung " + self.objective_dict[objective] + " ist " + label + "."
                    examples.append(InputExample(self.objective_dict[objective], sentence, text_b))

                start_index = len(features)
                features.extend(convert_examples_to_features(examples, self.max_seq_length, self.tokenizer))
                features_index[objective] = (start_index, len(features))

            all_features_index.append(features_index)
            all_input_ids.append(torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device))
            all_input_mask.append(torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device))

        return (all_input_ids, all_input_mask, all_features_index)

    def inference(self, inputs, **kwargs):
        """
        Predict the class of a text using a trained transformer model.
        :param **kwargs:
        """
        all_input_ids, all_input_mask, all_features_index = inputs

        predictions = []

        for input_ids, input_mask, features_index in zip(all_input_ids, all_input_mask, all_features_index):
            pred_dict = {}
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=input_mask)
                logits = F.softmax(outputs[0], dim=1)[:, 1].detach().cpu().numpy()

            for feature in features_index.keys():
                temp = softmax(logits[features_index[feature][0]: features_index[feature][1]])
                i = int(np.argmax(temp))
                pred_dict[feature] = {self.label_dict[list(self.label_dict.keys())[i]]: float(temp[i])}
            predictions.append(pred_dict)

        return predictions


    def postprocess(self, predictions):
        postprocess_output = {}
        for i, prediction in enumerate(predictions):
            postprocess_output["sent{}".format(i)] = {"text": self.sentences[i], "pred": prediction}

        return [json.dumps(postprocess_output)]


_service = TransformersClassifierHandler()


def handle(data, context):
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        data = _service.preprocess(data)
        data = _service.inference(data, )
        data = _service.postprocess(data)

        return data
    except Exception as e:
        raise e