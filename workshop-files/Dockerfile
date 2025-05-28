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

# 2 - Notebooks layer - add .ipynb and .py files
FROM base AS notebooks

WORKDIR /opt/intersystems

COPY data /opt/intersystems/data
COPY ["1. Embed Data Manually.ipynb", "/opt/intersystems/"]
COPY ["2. Building a Simple Chat.py", "/opt/intersystems/"]
COPY ["2a. Running Application.ipynb", "/opt/intersystems/"]
COPY ["3. Using Helper Libraries.ipynb", "/opt/intersystems/"]
COPY ["3b. Test-First Framework.ipynb", "/opt/intersystems/"]
COPY ["4. Connecting Chat to Vectors.py", "/opt/intersystems/"]
COPY ["4a. Running Application.ipynb", "/opt/intersystems/"]
COPY ["5. Putting it Together.py", "/opt/intersystems/"]
COPY ["5a. Running Application.ipynb", "/opt/intersystems/"]

CMD start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''