# pioneer_metals_rag_api

---

## Prerequisite

Set up a HuggingFace token at [HuggingFace](https://huggingface.co/settings/tokens)

## Installation

Make a virtual environment. From the project directory root run:

```
# generate the conda environment
conda create --name rag_api
conda activate rag_api
conda install -c python=3.12 conda-forge flask-cors langchain langchain-community langchain-chroma langchain-huggingface pypdf
```

## Running

Save the contents:

    export HUGGINGFACEHUB_API_TOKEN=$HF_TOKEN

where `HF_TOKEN` is the value of the API key from HuggingFace into a file called `hf_token.sh`.

Launch:

    ./launch.sh

