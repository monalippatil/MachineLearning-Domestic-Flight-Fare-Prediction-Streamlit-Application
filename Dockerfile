FROM jupyter/tensorflow-notebook:python-3.9.13

COPY requirements.txt .

RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/tf/notebooks"

WORKDIR /tf/notebooks
