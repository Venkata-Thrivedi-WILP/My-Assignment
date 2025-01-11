RUN -> python src/train.py


mlflow ui

docker build -t iris-model-flask .

docker run -p 5000:5000 iris-model-flask


input :

{
  "input": [5.1, 3.5, 1.4, 0.2]
}
