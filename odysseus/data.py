import json
import os
import random
import textwrap
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


def build_story_question_dict(data_path):
    data_dict = defaultdict(dict)
    count = 0
    for line in open(data_path, "r"):
        cols = line.strip().split("\t")
        data_dict[count]["story"] = cols[0]
        data_dict[count]["q1"] = cols[1]
        data_dict[count]["q2"] = cols[2]
        count += 1
    return data_dict


def build_answer_dict(data_path):
    name_dict = {}
    answers_dict = {}
    score_dict = {}
    ID_dict = {}

    if "2024_11_04" in data_path:
        xl_file = pd.ExcelFile(data_path)
        df_dict = {sheet_name: xl_file.parse(sheet_name) for sheet_name in xl_file.sheet_names if "Story" in sheet_name}  # type: ignore

        for ID, story_name in enumerate(df_dict.keys()):
            answers_ls = df_dict[story_name]["Unnamed: 3"].values.tolist()[2:]
            answers_ls_pad = []
            for line in answers_ls:
                if isinstance(line, str):
                    answers_ls_pad.append(line.strip())
                else:
                    answers_ls_pad.append("I dont know")

            student_ids = df_dict[story_name]["Unnamed: 0"].values.tolist()[2:]
            scores = df_dict[story_name]["Unnamed: 4"].values.tolist()[2:]

            name_dict[ID] = story_name
            ID_dict[ID] = student_ids
            answers_dict[ID] = answers_ls_pad
            score_dict[ID] = scores

    return name_dict, ID_dict, answers_dict, score_dict


# def build_nest_dict(data_path):
#     if "grading_scheme.txt" in data_path:
#         cols = open(data_path).read().split("\n\n")[1:]
#         example_dict = {ID: "\n".join(V.split("\n")[1:]) for ID, V in enumerate(cols)}
#     else:
#         example_dict = defaultdict(lambda: defaultdict(list))
#         doc = open(data_path).read()
#         group_ls = doc.split("\n\n")
#         for ID, group in enumerate(group_ls[1:]):
#             for line in group.split("\n"):
#                 if "\t" not in line:
#                     label = line.strip()[:-1]
#                 else:
#                     example_dict[ID][label].append(line.strip())
#     return example_dict


def build_coding_scheme(data_path):
    scheme_dict = {}
    all_files = sorted(
        [f for f in os.listdir(data_path) if f.endswith(".txt")],
        key=lambda x: int(x.split(".")[0]),
    )
    for ID, file in enumerate(all_files):
        file = os.path.join(data_path, file)
        text = open(file).read()
        scheme_dict[ID] = text
    return scheme_dict


def build_fewshot_xml(dic):
    prompt_template = textwrap.dedent("""
    <example>
    Answer: {content}
    Grade: {grade}
    </example>""")
    label2score = {"Incorrect": str(0), "Correct": str(1)}
    fewshot = [prompt_template.format(content=v, grade=label2score[k]) for k, vs in dic.items() for v in vs]
    random.shuffle(fewshot)
    prompt = "\n".join(fewshot)
    return prompt


def build_scheme_xml_format(dic):
    prompt_template = "<GradingScheme>" + "\n" + "{content}" + "\n" + "</GradingScheme>"
    xml = ""
    ls = [(k, v) for k, v in dic.items()]
    for line in ls:
        label, values = line
        out_str = "\t" + "<" + label.strip() + ">" + "\n"
        xml += out_str
        for v in values:
            out_str = "\t\t" + "<" + v.strip() + ">" + "\n"
            xml += out_str
        out_str = "\t" + "</" + label.strip() + ">" + "\n"
        xml += out_str
    prompt = prompt_template.format(content=xml)
    return prompt


def generate_split_indices(n_samples, test_val_size=0.10, random_state=42):
    idx = {}
    indices = list(range(n_samples))
    idx_trainval, idx["test"] = train_test_split(indices, test_size=test_val_size, random_state=random_state)
    idx["train"], idx["valid"] = train_test_split(idx_trainval, test_size=test_val_size, random_state=random_state)
    return idx


def format_dataset(id_2_name, id_2_storyquestion, id_2_scheme, id_2_answers, id_2_scores):
    def format_prompt(story, question, grading_scheme, answer, template):
        prompt = template.format(
            story=story.strip(),
            question=question.strip(),
            grading_scheme=grading_scheme.strip(),
            answer=answer.strip(),
        )
        return prompt

    # Use textwrap.dedent to remove leading whitespaces
    instruction = textwrap.dedent("""
    You are an assistant tasked with grading answers to a mind reading ability test. You will be provided with the following information:

    1. A story that was presented to participants as context
    2. The question that participants were asked to answer
    3. A grading scheme to evaluate the answers (Correct Responses:1, incorrect response:0, Incomplete response:0, Irrelevant:0)
    4. Grading examples
    5. A participant answer

    Your task is to grade each answer according to the grading scheme. For each answer, you should:

    1. Carefully read and understand the answer and compare it to the grading criteria
    2. Assigning an score 1 or 0 for each answer.
    """)

    prompt_template = textwrap.dedent("""
    <Story>
    {story}
    </Story>

    <Question>
    {question}
    </Question>

    <GradingScheme>
    {grading_scheme}
    </GradingScheme>

    <Answer>
    {answer}
    </Answer>

    Score:""")

    ds = []

    for story_id in id_2_name:
        story, _, question = id_2_storyquestion[story_id].values()
        grading_scheme = id_2_scheme[story_id]

        for student_id in range(len(id_2_answers[0])):
            answer = id_2_answers[story_id][student_id]
            # NOTE: Adding SYSTEM_PROMPT as 'instruction' here
            X = instruction.strip() + "\n" + format_prompt(story, question, grading_scheme, answer, prompt_template)
            y = id_2_scores[story_id][student_id]
            ds.append({"text": X, "label": y})
    return ds


def tokenize_dataset(dataset: DatasetDict, tokenizer, truncation, padding, max_length: int) -> DatasetDict:
    def prepare_for_chat_template(batch):
        batch["messages"] = [[{"role": "user", "content": text}] for text in batch["text"]]
        return batch

    def apply_chat_template(batch, tokenizer):
        # https://github.com/huggingface/alignment-handbook/blob/7d711cd80dff17df78f8340df1b7616af0809407/src/alignment/data.py#L48-L56
        batch["text"] = tokenizer.apply_chat_template(batch["messages"], add_generation_prompt=True, tokenize=False)
        return batch

    def tokenize(batch, tokenizer, truncation="max_length", padding="do_not_pad", max_length=1024):
        # Optimise map: https://discuss.huggingface.co/t/dataset-map-function-takes-forever-to-run/35694
        tokenized = tokenizer(
            batch["text"],
            truncation=True if truncation == "max_length" else truncation,
            padding=padding,
            max_length=max_length,
            return_tensors="np",
            # Avoid double BOS tokens
            # https://x.com/danielhanchen/status/1789659394302718373
            # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/discussions/97#66c3f47f96583c59b07270fa
            add_special_tokens=False,
        )
        return {**batch, **tokenized}

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.chat_template is not None:
        dataset = dataset.map(prepare_for_chat_template, batched=True, batch_size=1024)
        dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer}, batched=True, batch_size=1024)

    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "truncation": truncation, "padding": padding, "max_length": max_length},
        batched=True,
        batch_size=1024,
    )

    return dataset


def build_paraphrased_data(paraphrased_data_path):
    paraphrased_id_2_answers, paraphrased_id_2_scores = {}, {}

    paraphrased_data = json.loads(open(paraphrased_data_path, "r").read())
    for story in paraphrased_data:
        # NOTE: keys in paraphrased_answers JSON could be string
        _answers_and_scores = paraphrased_data.get(story) or paraphrased_data.get(f"{story}")
        if isinstance(story, str):
            story = int(story)
        paraphrased_id_2_answers[story], paraphrased_id_2_scores[story] = [], []
        for answer, score in _answers_and_scores:
            paraphrased_id_2_answers[story].append(answer)
            paraphrased_id_2_scores[story].append(score)

    return paraphrased_id_2_answers, paraphrased_id_2_scores


STORY_QUESTION_FILE = "story_question.txt"
ANSWER_FILE = "2024_11_04_Coded_Responses_for_1300_Online_Participants_&_400_School_and_College_Students.xlsx"
PARAPHRASED_DATA_FILE = "paraphrased_train_data_with_label.json"
CODING_SCHEME_DIR = "cleaned_coding_scheme_1909"


def add_paraphrased_data(
    dataset: DatasetDict,
    data_dir="./data",
    coding_scheme_dir=CODING_SCHEME_DIR,
    story_question_path=None,
    answer_path=None,
    paraphrased_data_path=None,
):
    assert "train" in dataset.keys(), "No trainset found. Have a 'train' key in your DatasetDict."

    print(f"Loading storyquestion & scheme from {data_dir}")
    if story_question_path is None:
        story_question_path = os.path.join(data_dir, STORY_QUESTION_FILE)
    if answer_path is None:
        answer_path = os.path.join(data_dir, ANSWER_FILE)
    if paraphrased_data_path is None:
        paraphrased_data_path = os.path.join(data_dir, PARAPHRASED_DATA_FILE)
    path_scheme = os.path.join(data_dir, coding_scheme_dir)

    id_2_storyquestion = build_story_question_dict(story_question_path)
    id_2_scheme = build_coding_scheme(path_scheme)
    id_2_name, _, _, _ = build_answer_dict(answer_path)
    paraphrased_id_2_answers, paraphrased_id_2_scores = build_paraphrased_data(paraphrased_data_path)

    print("Formatting data")
    paraphrased_data = format_dataset(
        id_2_name,
        id_2_storyquestion,
        id_2_scheme,
        paraphrased_id_2_answers,
        paraphrased_id_2_scores,
    )
    paraphrased_dataset = Dataset.from_list(paraphrased_data)

    print("Adding new data to trainset of original data")
    dataset["train"] = concatenate_datasets([dataset["train"], paraphrased_dataset])

    return dataset


def generate_dataset(
    data_dir="./data",
    coding_scheme_dir=CODING_SCHEME_DIR,
    story_question_path=None,
    answer_path=None,
    # split options
    test_val_size=0.1,
    seed=42,
    force=False,
    # tokenizer options
    model_id=None,
    save_dir="./results",
    truncation="max_length",
    padding="do_not_pad",
    max_length=1024,
    remove_unused_columns=False,
):
    # Test if dataset has already been generated if `force` is not set
    if save_dir is not None and not force:
        try:
            _ = load_dataset(save_dir)
            print(f"Dataset already exists at {save_dir}. Use --force to overwrite.")
            return
        except FileNotFoundError:
            pass

    if story_question_path is None:
        story_question_path = os.path.join(data_dir, STORY_QUESTION_FILE)
    if answer_path is None:
        answer_path = os.path.join(data_dir, ANSWER_FILE)
    path_scheme = os.path.join(data_dir, coding_scheme_dir)

    # Load data
    print(f"Loading data from {data_dir}")
    id_2_storyquestion = build_story_question_dict(story_question_path)
    id_2_scheme = build_coding_scheme(path_scheme)
    id_2_name, id_2_student, id_2_answers, id_2_scores = build_answer_dict(answer_path)

    # Format dataset
    print("Formatting data")
    ds = format_dataset(id_2_name, id_2_storyquestion, id_2_scheme, id_2_answers, id_2_scores)

    # Balance dataset
    print("Balancing dataset")
    sampler = RandomUnderSampler(random_state=seed)
    resampled = sampler.fit_resample(np.arange(len(ds)).reshape(-1, 1), [x["label"] for x in ds])
    resampled_idx = resampled[0].squeeze().tolist()  # type: ignore
    assert isinstance(resampled_idx, list)
    ds = [ds[i] for i in resampled_idx]

    # Split dataset
    print(f"Splitting data with test & val fraction {test_val_size}")
    idx = {}
    indices = list(range(len(ds)))
    idx_trainval, idx["test"] = train_test_split(
        indices, test_size=test_val_size, random_state=seed, stratify=[d["label"] for d in ds]
    )
    idx["train"], idx["validation"] = train_test_split(
        idx_trainval,
        test_size=test_val_size,
        random_state=seed,
        stratify=[ds[i]["label"] for i in idx_trainval],
    )

    # Build DatasetDict
    dataset = DatasetDict(
        {
            "train": Dataset.from_list([ds[i] for i in idx["train"]]),
            "validation": Dataset.from_list([ds[i] for i in idx["validation"]]),
            "test": Dataset.from_list([ds[i] for i in idx["test"]]),
        }
    )

    # Tokenize
    if model_id is not None:
        print(f"Tokenizing data with tokenizer {model_id}")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = tokenize_dataset(dataset, tokenizer, truncation, padding, max_length)

    # Clean
    if remove_unused_columns:
        dataset = dataset.select_columns(["input_ids", "attention_mask", "label"])

    # Save
    if save_dir is not None:
        # Save dataset to disk
        # Create save_dir if it does not exist
        os.makedirs(save_dir, exist_ok=True)
        # https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict.save_to_disk
        dataset.save_to_disk(save_dir, num_proc=os.cpu_count())
        print(f"✨ Saved dataset to {save_dir}")
    else:
        print("✨Done")

    return dataset


def add_paraphrased_data_to_trainset_of_existing_dataset(
    dataset_dir: str,
    data_dir="./data",
    coding_scheme_dir=CODING_SCHEME_DIR,
    story_question_path=None,
    answer_path=None,
    paraphrased_data_path=None,
    save_dir="./data/dataset_paraphrased",
    force=False,
):
    if save_dir is not None and not force:
        try:
            _ = load_dataset(save_dir)
            print(f"Dataset already exists at {save_dir}. Use --force to overwrite.")
            return
        except FileNotFoundError:
            pass

    dataset = load_dataset(dataset_dir)
    assert isinstance(dataset, DatasetDict), "out of scope"
    dataset = add_paraphrased_data(
        dataset=dataset,
        data_dir=data_dir,
        coding_scheme_dir=coding_scheme_dir,
        story_question_path=story_question_path,
        answer_path=answer_path,
        paraphrased_data_path=paraphrased_data_path,
    )

    if save_dir is not None:
        # Save dataset to disk
        # Create save_dir if it does not exist
        os.makedirs(save_dir, exist_ok=True)
        # https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.DatasetDict.save_to_disk
        dataset.save_to_disk(save_dir, num_proc=os.cpu_count())
        print(f"✨ Saved dataset to {save_dir}")
    else:
        print("✨Done")

    return dataset


def load_and_tokenize_dataset(
    dataset_dir: str,
    tokenizer,
    dynamic_padding: bool = True,
    max_length: int = 4096,
) -> DatasetDict:
    dataset = load_dataset(dataset_dir)
    assert isinstance(dataset, DatasetDict), "dataset type is out of scope"
    dataset = tokenize_dataset(
        dataset,
        tokenizer,
        truncation="max_length",
        padding="do_not_pad" if dynamic_padding else "max_length",
        max_length=max_length,
    )
    dataset = dataset.with_format("torch")
    return dataset


def separate_testset_by_story(testset: Dataset, data_dir, story_question_path=None):
    if story_question_path is None:
        story_question_path = os.path.join(data_dir, STORY_QUESTION_FILE)

    id_2_storyquestion = build_story_question_dict(story_question_path)
    id_2_story = {k: v["story"] for k, v in id_2_storyquestion.items()}
    id_2_testdsidx = {}
    for story_id, story_text in id_2_story.items():
        # Store the indices of matching samples instead of the samples themselves
        id_2_testdsidx[story_id] = [idx for idx, sample in enumerate(testset) if story_text in sample["text"]]  # type: ignore
    return id_2_testdsidx


if __name__ == "__main__":
    parser = ArgumentParser()
    # data
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--coding-scheme-dir", type=str, default="cleaned_coding_scheme_1909")
    parser.add_argument("--answer-path", type=str, default=None)
    parser.add_argument("--story-question-path", type=str, default=None)

    # split
    parser.add_argument("--test-val-size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    # Set the following options if you want to pre-tokenize the dataset
    # If model_id is None, no tokenization is performed
    # https://huggingface.co/docs/transformers/en/pad_truncation
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--truncation", default="do_not_truncate")
    parser.add_argument("--padding", default="do_not_pad")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--remove-unused-columns", action="store_true")

    # save
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--save-dir", type=str, default="./data/dataset")

    # misc
    parser.add_argument("--hf-token", type=str, default=None)

    args = parser.parse_args()

    if args.hf_token is not None:
        os.environ["HF_TOKEN"] = args.hf_token

    # Avoid tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    generate_dataset(
        data_dir=args.data_dir,
        coding_scheme_dir=args.coding_scheme_dir,
        answer_path=args.answer_path,
        story_question_path=args.story_question_path,
        model_id=args.model_id,
        test_val_size=args.test_val_size,
        save_dir=args.save_dir,
        truncation=args.truncation,
        padding=args.padding,
        max_length=args.max_length,
        seed=args.seed,
        force=args.force,
    )
