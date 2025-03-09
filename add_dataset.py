import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict


def load_jsonl_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data


def preprocess_dataset(dataset):
    formatted_data = []
    for sample in dataset:
        instruction = sample.get("instruction", "").strip()
        response = sample.get("response", "").strip()

        if instruction and response:
            formatted_data.append({"text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}"})

    return formatted_data


model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


dataset_path = "./code-end-to-end-cyber.jsonl"
dataset = load_jsonl_dataset(dataset_path)
processed_dataset = preprocess_dataset(dataset)


hf_dataset = Dataset.from_list(processed_dataset)


split_dataset = hf_dataset.train_test_split(test_size=0.1)  # 90% train, 10% eval


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = split_dataset.map(tokenize_function, batched=True)


training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    eval_strategy="epoch", 
    save_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)


data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    processing_class=tokenizer.__class__,  # ✅ Sesuai versi terbaru
    data_collator=data_collator,
  
)



trainer.train()


model.save_pretrained("./fine-tuned-deepseek-coder")
tokenizer.save_pretrained("./fine-tuned-deepseek-coder")
