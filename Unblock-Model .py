# Install dependencies
!pip install -q transformers datasets accelerate torch peft huggingface_hub
!pip install -q -U torch torchvision torchaudio
!pip install -q bitsandbytes
!pip install -q wandb

# Import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import json
import os
from huggingface_hub import login

# 🔹 Masukkan Token Hugging Face
HF_TOKEN = ""  # Ganti dengan token API Hugging Face kamu
login(HF_TOKEN)

# 🔹 Load Model Base
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

# 🔹 Atur pad_token agar tidak error saat padding
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 🔹 Hilangkan Blokir Kata di Tokenizer (Agar Uncensored)
blocked_words = ["hack", "exploit", "malware", "bypass security", "phishing"]
tokenizer.add_tokens(blocked_words)  # Tambahkan kata-kata yang sebelumnya diblokir
tokenizer.save_pretrained("custom_tokenizer")  # Simpan tokenizer yang telah diperbarui

# 🔹 Load Model dengan 4-bit Quantization (Lebih Ringan)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Gunakan float16 untuk menghemat VRAM
    device_map="auto"
)

# 🔹 Perbarui token embedding setelah menambahkan pad_token & blocked words
model.resize_token_embeddings(len(tokenizer))

# 🔹 Load Dataset dari Hugging Face
dataset = load_dataset("WhiteRabbitNeo/Code-Functions-Level-Cyber", token=HF_TOKEN)

# 🔹 Fungsi Tokenisasi Dataset
def tokenize_function(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"
    tokenized = tokenizer(prompt, padding="max_length", truncation=True, max_length=256)

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize_function, remove_columns=["instruction", "response"])

# 🔹 Konfigurasi LoRA untuk Fine-Tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05
)
model = get_peft_model(model, lora_config)

# 🔹 Konfigurasi Training
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama3_uncensored",
    per_device_train_batch_size=,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    remove_unused_columns=False,
    fp16=True,  # Aktifkan mixed precision untuk efisiensi di GPU
    report_to="none"  # Matikan logging ke W&B jika tidak digunakan
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 🔹 Trainer untuk Fine-Tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)

# 🔹 Mode Training & Jalankan Fine-Tuning
model.train()
trainer.train()

# 🔹 Simpan Model yang Sudah Dilatih
trainer.model.save_pretrained("fine_tuned_llama3_uncensored")
tokenizer.save_pretrained("fine_tuned_llama3_uncensored")

print("✅ Model selesai dilatih dan disimpan tanpa sensor!")

# 🔹 Modifikasi config.json agar tidak ada filter
config_path = "fine_tuned_llama3_uncensored/config.json"
with open(config_path, "r") as f:
    config_data = json.load(f)

# 🔹 Ubah agar model menerima semua kode tanpa sensor
config_data["trust_remote_code"] = True
with open(config_path, "w") as f:
    json.dump(config_data, f, indent=4)

print("✅ Konfigurasi model diubah agar lebih bebas.")
