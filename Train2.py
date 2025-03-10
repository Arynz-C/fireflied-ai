

# Impor pustaka yang diperlukan
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import torch
import json
import os
from huggingface_hub import login

# Token API Hugging Face untuk mengakses model
TOKEN_HF = "YOUR_HF_TOKEN"  # Ganti dengan token API milikmu
login(TOKEN_HF)

# Ambil Model Dasar dari Hugging Face
nama_model = "meta-llama/Llama-3.2-1B"
penyama_katacik = AutoTokenizer.from_pretrained(nama_model, token=TOKEN_HF)
print("✅ Tokenizer berhasil dimuat.")

# Atur pad_token agar tidak terjadi error saat padding
penyama_katacik.pad_token = penyama_katacik.eos_token
penyama_katacik.add_special_tokens({'pad_token': '[PAD]'})
print("✅ Tokenizer diperbarui dengan pad_token.")

# Muat model dasar dan tentukan jenis precision untuk efisiensi VRAM
modelnyacik = AutoModelForCausalLM.from_pretrained(
    nama_model,
    torch_dtype=torch.float16,  # Gunakan float16 agar lebih ringan dalam penggunaan memori
    device_map="auto"  # Secara otomatis memilih perangkat terbaik (GPU jika tersedia)
)
print("✅ Model berhasil dimuat.")

# Perbarui token embedding setelah menambahkan pad_token
modelnyacik.resize_token_embeddings(len(penyama_katacik))
print("✅ Token embedding diperbarui.")

# Ambil Dataset dari Hugging Face untuk pelatihan
dataset_latihan = load_dataset("WhiteRabbitNeo/Code-Functions-Level-Cyber", token=TOKEN_HF)
print("✅ Dataset berhasil dimuat.")

# Fungsi untuk Tokenisasi Data agar model dapat memproses input dengan benar
def proses_tokenisasi(contoh):
    perintah = f"### Instruksi:\n{contoh['instruction']}\n\n### Respon:\n{contoh['response']}"
    hasil_token = penyama_katacik(perintah, padding="max_length", truncation=True, max_length=256)
    hasil_token["labels"] = hasil_token["input_ids"].copy()  # Label harus sama dengan input_ids untuk fine-tuning
    return hasil_token

# Terapkan tokenisasi ke seluruh dataset
dataset_latihan = dataset_latihan.map(proses_tokenisasi, remove_columns=["instruction", "response"])
print("✅ Dataset selesai ditokenisasi.")

# Konfigurasi LoRA (Low-Rank Adaptation) untuk Fine-Tuning
konfig_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05
)
modelnyacik = get_peft_model(modelnyacik, konfig_lora)
print("✅ LoRA berhasil dikonfigurasi.")

# Pengaturan Pelatihan dengan Hyperparameter yang Disesuaikan
pengaturan_latihan = TrainingArguments(
    output_dir="./model_llama3_tanpa_sensor",
    per_device_train_batch_size=4,  # Meningkatkan batch size untuk konvergensi lebih cepat
    gradient_accumulation_steps=8,
    num_train_epochs=10,  # Menambah jumlah epoch agar loss bisa turun lebih jauh
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=5,  # Logging lebih sering untuk pemantauan loss
    remove_unused_columns=False,
    fp16=True,
    learning_rate=5e-5,  # Menurunkan learning rate untuk stabilitas
    weight_decay=0.01,  # Regularisasi untuk mencegah overfitting
    report_to="none"
)
print("✅ Pengaturan pelatihan berhasil dibuat.")

# Data Collator untuk menangani padding dalam batch
pengelola_data = DataCollatorForSeq2Seq(penyama_katacik, model=modelnyacik)
print("✅ Data collator berhasil dikonfigurasi.")

# Trainer untuk menjalankan Fine-Tuning
pelatih = Trainer(
    model=modelnyacik,
    args=pengaturan_latihan,
    train_dataset=dataset_latihan["train"],
    data_collator=pengelola_data
)
print("✅ Trainer berhasil dikonfigurasi.")

# Mode Latihan & Jalankan Fine-Tuning
modelnyacik.train()
print("✅ Model dalam mode pelatihan.")
pelatih.train()
print("✅ Fine-Tuning selesai.")

# Simpan Model yang Sudah Dilatih
pelatih.model.save_pretrained("model_llama3_tanpa_sensor")
penyama_katacik.save_pretrained("model_llama3_tanpa_sensor")
print("✅ Model selesai dilatih dan disimpan tanpa sensor!")

# Modifikasi config.json agar tidak ada filter
lokasi_konfig = "model_llama3_tanpa_sensor/config.json"
with open(lokasi_konfig, "r") as f:
    data_konfig = json.load(f)

# Ubah agar model menerima semua kode tanpa sensor
data_konfig["trust_remote_code"] = True
with open(lokasi_konfig, "w") as f:
    json.dump(data_konfig, f, indent=4)

print("✅ Konfigurasi model diubah agar lebih bebas.")
