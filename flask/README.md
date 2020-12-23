# Deploy model with Flask

As everything is wrapped in docker this module can be deployed easily.
First download the saved model into the flask folder ([GDrive](https://drive.google.com/drive/folders/1Db2oH7dRG1_oRT6JRDXs4vk7VQ6B8ZYR?usp=sharing])) and than build the docker container by running the following command:

```
docker build -t lexitech-flask:latest . 
```

After the docker container was created you can run it locally by executing the following line:

```
docker run -d -m=8g -p 5000:5000 lexitech-flask 
```

If you want to deploy our container to AWS EC2 the following commands need to be entered ([AWS_CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) needs to be installed):
```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin XXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com
docker tag lexitech-flask:latest XXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/lexitech-flask:latest
docker push XXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/lexitech-flask:latest
```