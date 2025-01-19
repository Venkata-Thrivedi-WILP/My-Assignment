LINT

flake8 src/

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
  "input": [5.1, 3.5, 1.4, 0.2]
}

--------------------------------------------------------------------------

docker login

docker tag iris-model-flask 2023aa05203/test-assignment-repo:latest

docker push 2023aa05203/test-assignment-repo:latest


https://hub.docker.com/repository/docker/2023aa05203/test-assignment-repo/general

--------------------------------------------------------------------------