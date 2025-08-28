.PHONY: install data train run test docker-build docker-run


install:
    pip install --upgrade pip && pip install -r requirements.txt


data:
    python src/data/make_dataset.py


train:
    python src/train.py


run:
    uvicorn app.main:app --reload


test:
    pytest -q


docker-build:
    docker build -t churn-api .


docker-run:
    docker run -p 8000:8000 churn-api

#make install
#make data
#make train
#make run
#make test
#make docker-build
#make docker-run