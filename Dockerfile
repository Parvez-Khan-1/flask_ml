FROM python:3.6

MAINTAINER PARVEZ KHAN

ENV APP_DIR /app

ENV PYTHONPATH $PYTHONPATH:${APP_DIR}

WORKDIR ${APP_DIR}

ADD requirements.txt ${APP_DIR}/requirements.txt

RUN pip install -r ${APP_DIR}/requirements.txt

RUN echo ${WORKDIR}

RUN python -m spacy download en_core_web_sm

ADD . ${APP_DIR}

CMD python ${APP_DIR}/src/main.py