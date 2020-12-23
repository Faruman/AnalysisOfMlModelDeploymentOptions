# Individual Analysis Paper

## Introduction

While often a lot of thought is put into the frameworks and methods which are used to develop a machine learning model, the frameworks on which the model will run in production often come up short. However, they are as important as they provide the endpoint through which the model is consumed by the user and thus significantly influences the users experience. Therefore, in the following paper three frameworks for deploying PyTorch models will be analysed on the basis of a concrete example and their advantages and disadvantages will be shown.


## Evaluation of different deployment methods for machine learning models

### The model

The model which will be deployed via the different frameworks, was build by myself for LexiTech, a start up I am engaged with which tries to help restaurants to better understand customer feedback. The model does hereby not only analyse the sentiment of a review but is also able to predict which facet of the restaurant (e.g.: food, price, ambience, ...) was evaluated. The architecture of the model is based on the one proposed by Chi Sun et al. in their paper “Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence” and was implemented with PyTorch (Sun, Huang, & Qiu, 2019). Combined with a dataset of 4.3 million annotated german reviews, different models with slightly altered architectures were created and assessed to find the optimal parameter combination. A plot of this evaluation can be seen below:

picture

Hereby the best performance was delivered by a rather large model (416 MB), based on the BERT architecture (Devlin, Chang, Lee, & Toutanova, 2019). Due to its size and complexity, the model seems to be a good fit to try out and evaluate different deployment techniques for machine learning models. As for most natural language processing applications, extensive pre-processing is required for the model to work. Thus, leading to the processing flow shown below:

picture

As can be seen not only tokenization needs to be applied to our text before applying the model, but different auxiliary sentences need to generate. This additional layer of complexity for the deployment of the example model, should be helpful for evaluating the flexibility of the different deployment frameworks.


### Flask

Flask is a lightweight WSGI web application framework. It is designed to make getting started quick and easy but is also flexible enough to scale up to complex applications. Due to these features, it has become one of the most popular Python web application frameworks (palletsprojects, 2020). Before the model can be deployed, a wrapper needs to build around it to handle the pre- and post-processing. This was done by creating a new class ([ABSApredictor](https://github.com/Faruman/CompIndividualAnalysisPaper/blob/master/flask/MyPredictor/predictors.py)), which will be called during the request to the flask API. After this is done, the flask server is set up with only a few lines of code:

```Python
import nltk
nltk.download('punkt')
import torch
from transformers import BertTokenizer
from MyPredictor import ABSApredictor

from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route('/home/', methods=['GET'])
@app.route("/", methods=['GET'])
def return_home():
    return "Space for the Lexitech API Documentation."

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
```

As can be seen, the code creates a post endpoint for text and will return the result to the user in json format. The easiest way to deploy this Flask web application is to wrap it inside a docker container, which can be easily deployed in a cloud environment with flexible scalability. The full description how to create the docker container and how to deploy it on AWS can be found within the [GitHub repository](https://github.com/Faruman/CompIndividualAnalysisPaper/tree/master/flask). 

To sum it up, Flask’s advantages are not only its simplicity which allows for a fast start of development, but also its flexibility which helps to implement APIs in the exact way they are needed. Furthermore, due to its popularity a lot of documentation exists and there is a strong, supportive community answering questions. 


### FastAPI

FastAPI is a modern, fast (high-performance), web framework, which is especially build for creating APIs. While sharing the simplicity and ease of use with Flask, it is not only much faster (up to 300%) but also has built in data validation and can handle asynchronous requests natively (tiangolo, 2020). The implementation process is similar to Flask and the setup of the server requires as few lines of code:

```Python
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
```

Furthermore, the deployment as docker is the same as for Flask as well (for details refer to the [GitHub repository](https://github.com/Faruman/CompIndividualAnalysisPaper/tree/master/fastapi)).

Due to FastAPI’s similarity to flask it also shares its advantages, namely the simplicity, as well as the flexibility. While fewer people are using FastAPI than Flask the documentation and the community support are still sufficient. However, FastAPI has additional advantages. For the [api test](https://github.com/Faruman/CompIndividualAnalysisPaper/blob/master/api_checker.py) it was not only 5% faster (FastAPI: 184.806 s, Flask: 196.029 s), but it is also able to detect invalid input data types at runtime by default. Furthermore, it automatically generates the documentation on the go when developing the API, saving developers work.

While improving on most of the limitations of Flask, creating APIs for additional models does not scale with FastAPI either.


### TorchServe

TorchServe is the result of a collaboration between Facebook and AWS and aims to provide a clean, well-supported, and industrial-grade path for deploying PyTorch models for inference at scale. As the library is part of the PyTorch open-source project, it is available for free (Spisak, Bindal, Chung, & Stefaniak, 2020). As the package is developed by Facebook, the framework is the officially supported way to deploy PyTorch models. Hereby the library does not only provide the system for generating the prediction, but also allows to serve multiple models simultaneously, version production models for A/B testing, load and unload models dynamically, and monitor detailed logs and metrics. Its structure can be seen in the graphic below:

Picture

As the example model does not use one of the custom handlers provided by PyTorch the first thing which needs to be done, is to create a custom handler. This handler implements different methods to initialize the model, preprocess the data, do the inference, and postprocess the prediction ([handler](https://github.com/Faruman/CompIndividualAnalysisPaper/blob/master/torchserve/handler.py)). Afterwards, a model archive (.mar file) can be created containing the model weights as well as the previously defined handler. Next, the model archive is deployed to the server and the Inference API can be used to make predictions. Furthermore, extensive logs are created, showing not only the amount of memory and cpu power consumed by the server, but also the time needed for each inference. Such a log entry can be seen below:

```
2020-12-23 15:38:16,909 - PredictionTime.Milliseconds:36505.0|#ModelName:bert_trimsweep,Level:Model|#hostname:DESKTOP-UHR4742,requestID:fc5b9f46-619a-44a5-8ed7-c67b1ff099c4,timestamp:1608734296
```

Due to the collaboration between AWS and Facebook for the creation of TorchServe, it is natively integrated with Amazon SageMaker and can therefore be easily deployed to the cloud. A detailed instruction on how to do this can be found in the [AWS documentation](https://aws.amazon.com/de/blogs/machine-learning/serving-pytorch-models-in-production-with-the-amazon-sagemaker-native-torchserve-integration/).

The clear advantage of this framework are the already existing features which are helpful when deploying a machine learning model for production, such as the integrated monitoring and logging. Furthermore, the framework allows to dynamically update and create APIs, making it ideal for production where service outages could be costly. Additionally, the possibility to easily add additional models makes it useful when working on a larger application. Moreover, the framework allows for parallel processing of requests making the framework more scalable than the other two alternatives. However, the additional features create an overhead, making TorchServe a bit slower than the other frameworks during the [api test](https://github.com/Faruman/CompIndividualAnalysisPaper/blob/master/api_checker.py) (236.411 s). Furthermore, due to the newness of the framework and its experimental status the documentation and community support is still limited.


## Conclusion

As we have seen, fastAPI might be preferable to flask in most of the cases as it provides the same simplicity and flexibility but adds additional features on top. Therefore, in the next step only FastAPI and TorchServe will be compared. In my opinion, FastAPI is preferable when deploying a proof of concept or creating an API for a limited number models, as it is simpler to handle and allows for more flexibility. However, as soon as multiple models should be deployed and dynamically managed as it is usually the case in production environments, TorchServe seems to be the way to go.

However, the three frameworks which were analysed and implemented represent only a first glimpse into the world of model deployment and additional options should be analysed. One solution which seems to be especially interesting is the deployment of models as AWS Lambda functions which allow them to run serverless.


