import torch
import torch.nn.functional as F
import numpy as np
from scipy.special import softmax


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


class ABSApredictor:
    def __init__(self, model, tokenizer, max_seq_length, device):
        self.device = device
        self.model = model.to(self.device)
        self.model = self.model.eval()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def predict(self, text_a):
        pred_dict= {}
        objective_dict = {"service": "des Services", "style": "des Stils", "food": "des Essens",
                          "quality": "der Qualität", "price": "des Preis", "restaurant": "des Restaurants",
                          "ambience": "des Ambientes", "drinks": "der Drinks", "location": "der Umgebung"}
        label_dict = {"positiv": "positive", "neutral": "neutral", "negativ": "negative", "uneinig": "conflict",
                      "unbestimmt": "none"}

        features = []
        features_index = {}

        for objective in objective_dict.keys():
            examples = []
            for label in label_dict.keys():
                text_b = "Die Bewertung " + objective_dict[objective] + " ist " + label + "."
                examples.append(InputExample(objective_dict[objective], text_a, text_b))

            start_index = len(features)
            features.extend(convert_examples_to_features(examples, self.max_seq_length, self.tokenizer))
            features_index[objective] = (start_index, len(features))

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(self.device)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(self.device)

        #experiment trying to reduce RAM usage
        #all_input = torch.stack((all_input_ids, all_input_mask), 2)
        #def apply_model(input_tensor):
        #    with torch.no_grad():
        #        return self.model(input_ids=input_tensor[:, 0].unsqueeze(0), attention_mask=input_tensor[:, 1].unsqueeze(0))
        #outputs = []
        #for i, one_input in enumerate(all_input):
        #    outputs.append(apply_model(one_input)[0])
        #outputs = torch.stack(outputs, 0).flatten(1)
        #logits = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()

        with torch.no_grad():
            outputs = self.model(input_ids=all_input_ids, attention_mask=all_input_mask)
            logits = F.softmax(outputs[0], dim=1)[:, 1].detach().cpu().numpy()

        for feature in features_index.keys():
            temp = softmax(logits[features_index[feature][0]: features_index[feature][1]])
            i = int(np.argmax(temp))
            pred_dict[feature] = {label_dict[list(label_dict.keys())[i]]: float(temp[i])}

        return pred_dict

if __name__ == '__main__':
    print("ABSApredictor module")