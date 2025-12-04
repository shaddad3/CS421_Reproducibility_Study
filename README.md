# **CS 421: Reproducibility Study for SusGen-GPT**

## Introduction 
In this study we setup and fine-tune a `Mistral 7B` model in a very similar way to how the author's of the **SusGen-GPT** paper do! We leverage Google Colab Pro's free monthly compute allowance of 100 CUs a month and their T4 GPU to work in a low compute environment (but one that at least has access to a GPU). Given the low resources, this training is sensitive in regards to memory and crashes or Out Of Memory errors. Also, Google Colab does not seem to gel well with `bitsandbytes` or with `QLoRA` implementations, and to get it to work you have to be on a earlier Python version (Python 3.10). However, Colab does not easily let you use a version that is that old, so lots of work arounds are needed and it is very finicky. 

## Requirements and How to Run
This repository contains a `requirements.txt` file produced by the authors which details all the installations needed to run the code. This `requirements.txt` file must be installed, and we do this inside of the notebook titled `CS421_Reproducibility_Study.ipynb`. Another important note is setting up the environment. As the authors detail in their README.md file (which can be found if you navigate inside of the `SusGen` folder), it is important to be running Python 3.10 for this to work properly. Below is the environment setup code recommended by the authors.
```shell
git clone git@github.com:JerryWu-code/SusGen.git
cd SusGen/
conda create --name susgen python==3.10 -y
conda activate susgen
export VLLM_INSTALL_PUNICA_KERNELS=1
pip install -r requirements.txt
```
Running the `CS421_Reproducibility_Study.ipynb` notebook works as long as you get Colab to use the Python 3.10 version. To use the author's exact setup and training configurations, follow their README.md file inside of the `SusGen` folder!

## Where to Get the Dataset
The dataset used in this study is provided in two places. In the top level directory where this README.md file is, there is the `SusGen-10k.json` file, and inside of the `SusGen` folder it is present too. You can also download the full `SusGen-30k` dataset from the link provided in either the report for this study, or in the notebook (it is within one of the markdown cells describing the dataset loading code). 

## Acknowledgements
This reproducibility study would not have been possible without the great paper, code, and data prepared by the SusGen-GPT authors! Their work was really interesting and well done, and I benefited highly from being able to explore their research in this paper.

# The Original README.md from the SusGen-GPT Authors
Below I add the original README.md file that is in the official repository attached to the SusGen paper. 


## 🌿 Introduction
We present **SusGen-30K**, a meticulously curated instruction-tuning dataset for financial and ESG NLP domain; Then, we introduce **SusGen-GPT**, a suite of fine-tuned LLMs that achieve state-of-the-art performance across financial and ESG benchmarks with only 7–8B parameters; Besides, we propose **TCFD-Bench**, a benchmark designed to evaluate sustainability report generation, setting a new standard for model evaluation in this domain.


<details>
<summary><strong>Tasks Supported</strong> (Click to expand)</summary>

>Headline Classification (**HC**), Named Entity Recognition (**NER**), Relation Extraction (**RE**), Sentiment Analysis (**SA**), Financial Question Answering (**FIN-QA**), Financial Tabel Question Answering (**FIN-TQA**), Text Summarisation (**SUM**), Sustainability Report Generation (**SRG**).

</details>

## Getting Started

### 1. Set Up the Environment

```shell
git clone git@github.com:JerryWu-code/SusGen.git
cd SusGen/
conda create --name susgen python==3.10 -y
conda activate susgen
export VLLM_INSTALL_PUNICA_KERNELS=1
pip install -r requirements.txt
```

### 2. Download the model checkpoint
Before you downloaded the checkpoint, make sure you have access to [Llama3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct), [Llama3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B), [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) and [Mistral-7B-v0.3](https://huggingface.co/mistralai/Mistral-7B-v0.3)
 and our lora [checkpoint](https://huggingface.co/WHATX/). And login in your huggingface client in the terminal.
```shell
# 1. set up the huggingface client
huggingface-cli login # prompt in your write permit access token
mkdir ckpts && cd ckpts/
git lfs install
# 2. download the llm base checkpoint
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B
git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
git clone https://huggingface.co/mistralai/Mistral-7B-v0.3
# 3. download the lora checkpoint (replace the 'path-to-our-lora-checkpoint' with the actual path)
git clone https://huggingface.co/WHATX/path-to-our-lora-checkpoint
```
<!-- ### Run the demo

```shell
cd src/
CUDA_VISIBLE_DEVICES=0 python demo.py --base_model Mistral-7B-Instruct-v0.3 --lora_path ../ckpts/path-to-our-lora-checkpoint-dir
``` -->

### 3. Data Preparation

You could download:
1) Only the data **SusGen-30k** in our huggingface repository via this [link](https://huggingface.co/datasets/WHATX/SusGen-30k), and put it under the folder [data/SusGen/](./data/SusGen/).
2) Data along with preprocessing parts and additional data via this [link](https://huggingface.co/datasets/WHATX/SusGen). And put it under the folder [data/](./data/).

### 4. Training
Adjust configs in the `configs/training_configs/finetune_config.yaml` and then run the following command:
```python
cd src/
CUDA_VISIBLE_DEVICES=0 /home/(you hostname)/anaconda3/envs/susgen/bin/torchrun --nproc_per_node=1 --master_port=29501 finetune.py --config configs/training_configs/finetune_config.yaml
```

### 5. Evaluation
```python
cd eval/code/
CUDA_VISIBLE_DEVICES=0 python eval.py
```
