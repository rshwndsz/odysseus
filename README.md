[_Odysseus_](https://en.wikipedia.org/wiki/Odysseus) - navigating the minds of others is its own epic journey.

## Automatic Scoring of an Open-Response Measure of Advanced Mind-Reading Using LLMs

A rigorous psychometric approach is crucial for the accurate measurement of mind-reading abilities. 
Traditional scoring methods for such tests, which involve lengthy free-text responses, require considerable time and human effort. 
This study investigates the use of large language models (LLMs) to automate the scoring of psychometric tests. 
Data were collected from participants aged 13 to 30 years and scored by trained human coders to establish a benchmark. 
We evaluated multiple LLMs against human assessments, exploring various prompting strategies to optimize performance and fine-tuning the models using a subset of the collected data to enhance accuracy. 
Our results demonstrate that LLMs can assess advanced mind-reading abilities with over 90\% accuracy on average. 
Notably, in most test items, the LLMs achieved higher Kappa agreement with the lead coder than two trained human coders, highlighting their potential to reliably score open-response psychometric tests.

>[!NOTE]
> Read our paper at [https://aclanthology.org/2025.clpsych-1.7](https://aclanthology.org/2025.clpsych-1.7)

## Dataset
| Description | Link |
| ------------- | ------------- |
| Base  | [🤗 rshwndsz/ToM-auto-scoring-base](https://huggingface.co/datasets/rshwndsz/ToM-auto-scoring-base) |

## Models

### Finetuned without paraphrasing 
| Base Model  | Link |
| ------------- | ------------- |
| allenai/longformer-base-4096  | [🤗 rshwndsz/ft-longformer-base-4096](https://huggingface.co/rshwndsz/ft-longformer-base-4096) |
| NousResearch/Hermes-3-Llama-3.2-3B  | [🤗 rshwndsz/ft-hermes-3-llama-3.2-3b](https://huggingface.co/rshwndsz/ft-hermes-3-llama-3.2-3b)  |
| microsoft/Phi-3.5-mini-instruct | [🤗 rshwndsz/ft-mistral-7b-v0.3-instruct](https://huggingface.co/rshwndsz/ft-mistral-7b-v0.3-instruct) |
| mistralai/Mistral-7B-v0.3-Instruct | [🤗 rshwndsz/ft-phi-3.5-mini-instruct](https://huggingface.co/rshwndsz/ft-phi-3.5-mini-instruct) |
| microsoft/phi-4 | [🤗 rshwndsz/ft-phi-4](https://huggingface.co/rshwndsz/ft-phi-4) |

### Finetuned with paraphrasing
| Base Model  | Link |
| ------------- | ------------- |
| allenai/longformer-base-4096  | [🤗 rshwndsz/ft_paraphrased-longformer-base-4096](https://huggingface.co/rshwndsz/ft_paraphrased-longformer-base-4096) |
| NousResearch/Hermes-3-Llama-3.2-3B  | [🤗 rshwndsz/ft_paraphrased-hermes-3-llama-3.2-3b](https://huggingface.co/rshwndsz/ft_paraphrased-hermes-3-llama-3.2-3b)  |
| microsoft/Phi-3.5-mini-instruct | [🤗 rshwndsz/ft_paraphrased-mistral-7b-v0.3-instruct](https://huggingface.co/rshwndsz/ft_paraphrased-mistral-7b-v0.3-instruct) |
| mistralai/Mistral-7B-v0.3-Instruct | [🤗 rshwndsz/ft_paraphrased-phi-3.5-mini-instruct](https://huggingface.co/rshwndsz/ft_paraphrased-phi-3.5-mini-instruct) |
| microsoft/phi-4 | [🤗 rshwndsz/ft_paraphrased-phi-4](https://huggingface.co/rshwndsz/ft_paraphrased-phi-4) |

## Usage

Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a new virtual environment 
```bash
uv venv .venv --python python3.10 --prompt odysseus
```

Change or create a new config file in `experiments`.

To benchmark zero-shot performance

```console
(odysseus) $ chmod +x scripts/zeroshot.sh
(odysseus) $ ./scripts/zeroshot.sh
```

To run a hyperparameter search
```console
(odysseus) $ wandb sweep --project odysseus sweep_phi4.yaml
(odysseus) $ chmod +x scripts/hpsearch.sh
(odysseus) $ ./scripts/hpsearch.sh
```

To run a finetune
```console
(odysseus) $ chmod +x scripts/finetune.sh
(odysseus) $ ./scripts/finetune.sh 
```

To compute per-story metrics
```console
(odysseus) $ chmod +x scripts/metrics.sh
(odysseus) $ ./scripts/metrics.sh
```


## Citation
To cite this work, please use the following BibTeX entry:
```bibtex
@inproceedings{wang-etal-2025-automatic,
    title = "Automatic Scoring of an Open-Response Measure of Advanced Mind-Reading Using Large Language Models",
    author = "Wang, Yixiao  and
      Dsouza, Russel  and
      Lee, Robert  and
      Apperly, Ian  and
      Devine, Rory  and
      van der Kleij, Sanne  and
      Lee, Mark",
    editor = "Zirikly, Ayah  and
      Yates, Andrew  and
      Desmet, Bart  and
      Ireland, Molly  and
      Bedrick, Steven  and
      MacAvaney, Sean  and
      Bar, Kfir  and
      Ophir, Yaakov",
    booktitle = "Proceedings of the 10th Workshop on Computational Linguistics and Clinical Psychology (CLPsych 2025)",
    month = may,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.clpsych-1.7/",
    doi = "10.18653/v1/2025.clpsych-1.7",
    pages = "79--89",
    ISBN = "979-8-89176-226-8",
    abstract = "A rigorous psychometric approach is crucial for the accurate measurement of mind-reading abilities. Traditional scoring methods for such tests, which involve lengthy free-text responses, require considerable time and human effort. This study investigates the use of large language models (LLMs) to automate the scoring of psychometric tests. Data were collected from participants aged 13 to 30 years and scored by trained human coders to establish a benchmark. We evaluated multiple LLMs against human assessments, exploring various prompting strate- gies to optimize performance and fine-tuning the models using a subset of the collected data to enhance accuracy. Our results demonstrate that LLMs can assess advanced mind-reading abilities with over 90{\%} accuracy on average. Notably, in most test items, the LLMs achieved higher Kappa agreement with the lead coder than two trained human coders, highlighting their potential to reliably score open-response psychometric tests."
}
```
