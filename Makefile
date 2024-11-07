deploy: build tag push
build:
	docker build --platform linux/amd64 -t capital1/lambda1 .
tag:
	docker tag capital1/lambda1:latest 412381780180.dkr.ecr.us-east-2.amazonaws.com/capital1/lambda1:latest
push:
	docker push 412381780180.dkr.ecr.us-east-2.amazonaws.com/capital1/lambda1:latest
