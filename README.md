# ToM-automatic-scoring-using-LLMs

A rigorous psychometric approach is crucial for the accurate measurement of mind-reading abilities. Traditional scoring methods for such tests, which involve lengthy free-text responses, require considerable time and human effort. This study investigates the use of large language models (LLMs) to automate the scoring of psychometric tests. Data were collected from participants aged 13 to 30 years and scored by trained human coders to establish a benchmark. We evaluated multiple LLMs against human assessments, exploring various prompting strategies to optimize performance and fine-tuning the models using a subset of the collected data to enhance accuracy. Our results demonstrate that LLMs can assess advanced mind-reading abilities with over 90\% accuracy on average. Notably, in most test items, the LLMs achieved higher Kappa agreement with the lead coder than two trained human coders, highlighting their potential to reliably score open-response psychometric tests.

## Dataset
| Base Model  | Link |
| ------------- | ------------- |
| Base  | [🤗 rshwndsz/ToM-auto-scoring-base](https://huggingface.co/datasets/rshwndsz/ToM-auto-scoring-base) |
| With Paraphrasing  | [🤗 rshwndsz/ToM-auto-scoring-paraphrased](https://huggingface.co/datasets/rshwndsz/ToM-auto-scoring-paraphrased)  |

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
