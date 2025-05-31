# Dockerfile

# 1 - Base image: install Jupyter + Python dependencies
FROM jupyter/minimal-notebook:latest AS base

WORKDIR /opt/intersystems

RUN \
  jupyter labextension disable "@jupyterlab/apputils-extension:announcements" && \
  pip install \
    langchain-iris \
    python-dotenv \
    ipython-sql \
    sqlalchemy-iris \
    irissqlcli \
    ipywidgets \
    datasets \
    jsonlines \
    huggingface_hub \
    jq \
    langchain-openai \
    langchain_community \
    sentence_transformers \
    streamlit \
    urlextract \
    fastembed \
    matplotlib \
    seaborn \
    ragas==0.1.7 \
    pandas

CMD start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''