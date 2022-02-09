FROM python:3.8

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r ./requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/"

# Add local optabm - for now
ADD /optabm /optabm

RUN pip install /optabm

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

ENV PYTHONPATH "${PYTHONPATH}:/"
