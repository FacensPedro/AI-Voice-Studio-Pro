#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_voice.py

Treinamento “real” do pipeline de clonagem de voz usando o
repositório Real-Time-Voice-Cloning (https://github.com/CorentinJ/Real-Time-Voice-Cloning).

Dependências:
    - Clone este repositório:
        git clone https://github.com/CorentinJ/Real-Time-Voice-Cloning.git
    - Instale dependências (no diretório raiz do RTVC):
        pip install -r requirements.txt
    - Certifique-se de que Python 3.8+ e CUDA (se houver GPU) estão configurados.

Uso:
    python train_voice.py \
        --dataset_root /caminho/para/seu/dataset \
        --encoder_epochs 100 \
        --synthesizer_epochs 1000 \
        --vocoder_epochs 250000
"""

import argparse
import os
import sys
import subprocess

def train_encoder(dataset_root, encoder_epochs):
    """
    Chama o script encoder_train.py do Real-Time-Voice-Cloning.
    """
    script = os.path.join("Real-Time-Voice-Cloning", "encoder", "encoder_train.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Não encontrou '{script}'. Verifique seu path do RTVC.")
    cmd = [
        sys.executable, script,
        "--datasets_root", dataset_root,
        "--save_path", "./encoder/saved_models",
        "--training_epochs", str(encoder_epochs)
    ]
    print(">> Iniciando treinamento do Encoder...")
    subprocess.check_call(cmd)


def train_synthesizer(dataset_root, synthesizer_epochs):
    """
    Chama o script synthesizer_train.py do Real-Time-Voice-Cloning.
    """
    script = os.path.join("Real-Time-Voice-Cloning", "synthesizer", "synthesizer_train.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Não encontrou '{script}'. Verifique seu path do RTVC.")
    cmd = [
        sys.executable, script,
        "--datasets_root", dataset_root,
        "--save_path", "./synthesizer/saved_models",
        "--n_epochs", str(synthesizer_epochs)
    ]
    print(">> Iniciando treinamento do Synthesizer...")
    subprocess.check_call(cmd)


def train_vocoder(vocoder_epochs):
    """
    Chama o script vocoder_train.py do Real-Time-Voice-Cloning.
    """
    script = os.path.join("Real-Time-Voice-Cloning", "vocoder", "vocoder_train.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(f"Não encontrou '{script}'. Verifique seu path do RTVC.")
    # Para muitos datasets de Vocoder (por padrão, o RTVC usa o dataset 'VCTK' processado pelo Synthesizer),
    # o argumento principal é --model (WaveRNN, etc.) e os paths dos checkpoints.
    cmd = [
        sys.executable, script,
        "--save_path", "./vocoder/saved_models",
        "--n_epochs", str(vocoder_epochs)
    ]
    print(">> Iniciando treinamento do Vocoder...")
    subprocess.check_call(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento completo do pipeline RTVC (Encoder + Synthesizer + Vocoder).")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Pasta raiz do dataset (ex: VCTK ou LJSpeech) formatada para RTVC"
    )
    parser.add_argument(
        "--encoder_epochs",
        type=int,
        default=100,
        help="Número de épocas para treinar o Encoder (padrão: 100)"
    )
    parser.add_argument(
        "--synthesizer_epochs",
        type=int,
        default=1000,
        help="Número de épocas para treinar o Synthesizer (padrão: 1000)"
    )
    parser.add_argument(
        "--vocoder_epochs",
        type=int,
        default=250000,
        help="Número de iterações para treinar o Vocoder (padrão: 250000)"
    )

    args = parser.parse_args()

    # Confirmações básicas:
    if not os.path.isdir(args.dataset_root):
        print(f"[ERRO] '{args.dataset_root}' não é uma pasta válida.")
        sys.exit(1)

    # Chama as funções de treinamento em sequência:
    train_encoder(args.dataset_root, args.encoder_epochs)
    train_synthesizer(args.dataset_root, args.synthesizer_epochs)
    train_vocoder(args.vocoder_epochs)

    print(">> Treinamento concluído com sucesso! Confira as pastas 'encoder/saved_models', 'synthesizer/saved_models' e 'vocoder/saved_models'.")
