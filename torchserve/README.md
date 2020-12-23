# Deploy model with TorchServe

To use TorchServe at first a Model Archive of the model we want to deploy needs to be created. This is done by the following command:

```
torch-model-archiver --model-name "bert_trimsweep" --version 1.0 --serialized-file ./pytorch_model.bin --extra-files "./config.json" --handler "./handler.py"
```

This creates the file bert_trimsweep which needs to be moved to the model_store folder:

```
mkdir model_store
mv bert_trimsweep.mar ./model_store/bert_trimsweep.mar
```

Now, after everything is in place we can start our server:
```
torchserve --start --model-store model_store --models bert_trimsweep=bert_trimsweep.mar
```

