FROM python:3.9.7-slim

WORKDIR /app
RUN mkdir data

COPY "./data/" "./data/"

RUN pip install -U pip
RUN pip install pipenv

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy --ignore-pipfile

COPY [ "starter.py", "model.bin", "./"]

#CMD["pipenv","shell"]

RUN pipenv update

CMD ["pipenv","run","python","starter.py"]

