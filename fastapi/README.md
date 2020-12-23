# Deploy model with FastAPI

As everything is wrapped in docker this module can be deployed easily.
First download the saved model into the fastapi folder ([GDrive](https://drive.google.com/drive/folders/1Db2oH7dRG1_oRT6JRDXs4vk7VQ6B8ZYR?usp=sharing])) and than build the docker container by running the following command:

```
docker build -t lexitech-fastapi:latest . 
```

After the docker container was created we can run it locally by executing the following line:

```
docker run -d -m=8g -p 8000:8000 lexitech-fastapi
```

If you want to deploy our container to AWS EC2 the following commands need to be entered ([AWS_CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) needs to be installed):

```
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin XXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com
docker tag lexitech-fastapi:latest XXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/lexitech-fastapi:latest
docker push XXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/lexitech-fastapi:latest
```