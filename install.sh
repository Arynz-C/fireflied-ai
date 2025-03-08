#!/bin/bash

RAM_MIN=8  
CPU_MIN=2  
RAM_RECOMMENDED=16
CPU_RECOMMENDED=4

RAM_AVAILABLE=$(free -g | awk '/^Mem:/{print $2}')

CPU_AVAILABLE=$(nproc)

if [[ $RAM_AVAILABLE -lt $RAM_MIN || $CPU_AVAILABLE -lt $CPU_MIN ]]; then
    echo "[WARNING] Spesifikasi sistem tidak mencukupi. Minimum: ${RAM_MIN}GB RAM, ${CPU_MIN} Core CPU."
    echo "Untuk kinerja yang lebih baik, disarankan: ${RAM_RECOMMENDED}GB RAM, ${CPU_RECOMMENDED} Core CPU."
fi

echo "Apakah Anda ingin melanjutkan instalasi? (y/n)"
read -r pilihan
if [[ $pilihan != "y" ]]; then
    echo "Instalasi dibatalkan."
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo "[INFO] Mengunduh dan menginstal Ollama..."
    curl -fsSL https://ollama.com/install.sh | bash
else
    echo "[INFO] Ollama sudah terinstal."
fi

echo "[INFO] Menjalankan Ollama Serve..."
ollama serve &
sleep 5

echo "[INFO] Menjalankan model Fireflies..."
ollama run Arynz/FireFlies
