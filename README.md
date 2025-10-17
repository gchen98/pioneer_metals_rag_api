# pioneer_metals_rag_api

---

## Prerequisite

Set up a HuggingFace token at [HuggingFace](https://huggingface.co/settings/tokens)

## Installation

Make a virtual environment. From the project directory root run:

```
# generate the conda environment
conda env create -f environment.yml
conda activate rag_api
# add the OCR library DocTR
pip3 install python-doctr
```

## Running

Save the contents:

    export HUGGINGFACEHUB_API_TOKEN=$HF_TOKEN

where `HF_TOKEN` is the value of the API key from HuggingFace into a file called `hf_token.sh`.

Launch:

    ./launch.sh

