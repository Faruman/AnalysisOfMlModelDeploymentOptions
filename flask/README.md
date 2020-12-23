# Deploy model with Flask

As everything is wrapped in docker this module can be deployed easily.
First download the saved model into the flask folder and than run the following commands:

```
docker build -t lexitech-flask:latest . 
```

```
docker run -d -m=8g -p 5000:5000 lexitech-flask 
```
