from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch


model_path = "/workspaces/fireflied-ai/llama.cpp/deepseek-model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,  # Gunakan float32 karena CPU only flat 16 untuk gpu ygy
    device_map={"": "cpu"}      # Paksa model berjalan di CPU 
)


dataset = load_dataset("json", data_files="/workspaces/fireflied-ai/code-end-to-end-cyber.jsonl")


def tokenize_function(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=128) #bisa lu naikin tapi ya minimal punya sepekan nasa
    
    tokenized["labels"] = tokenized["input_ids"].copy()  # Labels harus sama dengan input_ids untuk causal LM
    return tokenized
    

dataset = dataset.map(tokenize_function, remove_columns=["instruction", "response"])


print("Contoh data setelah tokenisasi:")
print(dataset["train"][0])


lora_config = LoraConfig(
    r=8,                     # Rank LoRA (semakin besar, semakin kompleks) tapi sepek pc lu nasa
    lora_alpha=16,           # Scaling factor
    target_modules=["q_proj", "k_proj", "v_proj"],  # 🔹 Tambahkan "k_proj" dan "v_proj"
    lora_dropout=0.05        # Dropout untuk regulasi
)
model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir="./fine_tuned_deepseek",
    per_device_train_batch_size=1,  # CPU only, batch kecil agar tidak kehabisan RAM ,naikin=nambah penggunaan memori
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False  # Pastikan kolom yang diperlukan tidak dihapus
)


data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)


trainer.train()


trainer.model.save_pretrained("fine_tuned_deepseek")
tokenizer.save_pretrained("fine_tuned_deepseek")
print("✅ Model selesai dilatih dan disimpan!")
