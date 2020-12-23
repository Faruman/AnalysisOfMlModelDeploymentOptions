# Deploy model with Flask

As everything is wrapped in docker this module can be deployed easily.
First download the saved model into the flask folder ([https://drive.google.com/drive/folders/1Db2oH7dRG1_oRT6JRDXs4vk7VQ6B8ZYR?usp=sharing](GDrive])) and than run the following commands:

```
docker build -t lexitech-flask:latest . 
```

```
docker run -d -m=8g -p 5000:5000 lexitech-flask 
```
