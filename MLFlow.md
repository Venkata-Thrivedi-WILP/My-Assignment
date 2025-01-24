-----------------------------------------------
LINT

flake8 src/

------------------------------------------------

Pipelines are in .github folder

-------------------------------------------------

Load Dataset => python scripts/save_iris_dataset.py

RUN -> python src/train.py

---------------------------------------------------

pip install dvc

dvc init

dvc add data/iris.csv

dvc status

-- Add a line in dataset

dvc status

dvc add data/iris.csv

-- Commit chnages for iris.csv.dvc

dvc push

---------------------------------------------------

mlflow ui

---------------------------------------------------
docker build -t iris-model-flask .


docker run -p 5000:5000 iris-model-flask

---------------------------------------------------

input :

{
  "input": [7.4,2.8,6.1,1.9]
}

--------------------------------------------------------------------------

docker login

docker tag iris-model-flask 2023aa05203/my-repo:latest


docker push 2023aa05203/my-repo:latest


https://hub.docker.com/repository/docker/2023aa05203/test-assignment-repo/general

--------------------------------------------------------------------------