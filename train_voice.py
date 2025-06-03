#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py

Treinamento completo do pipeline de clonagem de voz usando o
Real-Time-Voice-Cloning (https://github.com/CorentinJ/Real-Time-Voice-Cloning).

Certifique-se de que:
  - O repositório Real-Time-Voice-Cloning está clonado no mesmo diretório deste script.
  - As dependências do RTVC foram instaladas: `pip install -r Real-Time-Voice-Cloning/requirements.txt`.
  - Python >= 3.8 e CUDA (se houver GPU) estão configurados corretamente.

Uso:
  python train.py \
    --dataset_root /caminho/para/o/dataset \
    [--encoder_epochs 100] \
    [--synthesizer_epochs 1000] \
    [--vocoder_epochs 250000]
"""

import argparse
import os
import sys
import subprocess

def ensure_directory(path: str):
    """
    Garante que um diretório exista. Se não existir, cria-o.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"[ERRO] Não foi possível criar o diretório '{path}': {e}")
        sys.exit(1)

def run_subprocess(cmd: list, cwd: str = None):
    """
    Executa um comando via subprocess.check_call, exibindo saída e capturando erros.
    """
    try:
        subprocess.check_call(cmd, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"[ERRO] Comando falhou (código {e.returncode}): {' '.join(cmd)}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"[ERRO] Executável não encontrado: {cmd[0]}")
        sys.exit(1)


def train_encoder(rtvc_root: str, dataset_root: str, encoder_epochs: int):
    """
    Invoca o script encoder_train.py do RTVC.
    """
    script_path = os.path.join(rtvc_root, "encoder", "encoder_train.py")
    if not os.path.isfile(script_path):
        print(f"[ERRO] Não encontrou '{script_path}'. Verifique o path do Real-Time-Voice-Cloning.")
        sys.exit(1)

    output_dir = os.path.join(rtvc_root, "encoder", "saved_models")
    ensure_directory(output_dir)

    cmd = [
        sys.executable, script_path,
        "--datasets_root", dataset_root,
        "--save_path", output_dir,
        "--training_epochs", str(encoder_epochs)
    ]

    print(">> Iniciando treinamento do Encoder...")
    run_subprocess(cmd, cwd=rtvc_root)
    print(">> Encoder treinado e salvo em:", output_dir)


def train_synthesizer(rtvc_root: str, dataset_root: str, synthesizer_epochs: int):
    """
    Invoca o script synthesizer_train.py do RTVC.
    """
    script_path = os.path.join(rtvc_root, "synthesizer", "synthesizer_train.py")
    if not os.path.isfile(script_path):
        print(f"[ERRO] Não encontrou '{script_path}'. Verifique o path do Real-Time-Voice-Cloning.")
        sys.exit(1)

    output_dir = os.path.join(rtvc_root, "synthesizer", "saved_models")
    ensure_directory(output_dir)

    cmd = [
        sys.executable, script_path,
        "--datasets_root", dataset_root,
        "--save_path", output_dir,
        "--n_epochs", str(synthesizer_epochs)
    ]

    print(">> Iniciando treinamento do Synthesizer...")
    run_subprocess(cmd, cwd=rtvc_root)
    print(">> Synthesizer treinado e salvo em:", output_dir)


def train_vocoder(rtvc_root: str, vocoder_epochs: int):
    """
    Invoca o script vocoder_train.py do RTVC.
    """
    script_path = os.path.join(rtvc_root, "vocoder", "vocoder_train.py")
    if not os.path.isfile(script_path):
        print(f"[ERRO] Não encontrou '{script_path}'. Verifique o path do Real-Time-Voice-Cloning.")
        sys.exit(1)

    output_dir = os.path.join(rtvc_root, "vocoder", "saved_models")
    ensure_directory(output_dir)

    cmd = [
        sys.executable, script_path,
        "--save_path", output_dir,
        "--n_epochs", str(vocoder_epochs)
    ]

    print(">> Iniciando treinamento do Vocoder...")
    run_subprocess(cmd, cwd=rtvc_root)
    print(">> Vocoder treinado e salvo em:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Treinamento completo do pipeline RTVC (Encoder + Synthesizer + Vocoder)."
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Pasta raiz do dataset (ex: VCTK, LJSpeech) no formato esperado pelo RTVC."
    )
    parser.add_argument(
        "--encoder_epochs",
        type=int,
        default=100,
        help="Número de épocas para treinar o Encoder (padrão: 100)."
    )
    parser.add_argument(
        "--synthesizer_epochs",
        type=int,
        default=1000,
        help="Número de épocas para treinar o Synthesizer (padrão: 1000)."
    )
    parser.add_argument(
        "--vocoder_epochs",
        type=int,
        default=250000,
        help="Número de iterações para treinar o Vocoder (padrão: 250000)."
    )

    args = parser.parse_args()

    dataset_root = args.dataset_root
    if not os.path.isdir(dataset_root):
        print(f"[ERRO] '{dataset_root}' não é uma pasta válida.")
        sys.exit(1)

    # Detecta o diretório do RTVC a partir da localização deste script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rtvc_root = os.path.join(script_dir, "Real-Time-Voice-Cloning")
    if not os.path.isdir(rtvc_root):
        print(f"[ERRO] Diretório Real-Time-Voice-Cloning não encontrado em: {rtvc_root}")
        sys.exit(1)

    print("=== Iniciando pipeline de treinamento RTVC ===")
    print(f"Dataset root: {dataset_root}")
    print(f"RTVC root: {rtvc_root}")
    print(f"Encoder epochs: {args.encoder_epochs}")
    print(f"Synthesizer epochs: {args.synthesizer_epochs}")
    print(f"Vocoder epochs: {args.vocoder_epochs}\n")

    # Treinamento sequencial
    train_encoder(rtvc_root, dataset_root, args.encoder_epochs)
    train_synthesizer(rtvc_root, dataset_root, args.synthesizer_epochs)
    train_vocoder(rtvc_root, args.vocoder_epochs)

    print("\n>> Treinamento concluído com sucesso!")
    print("Modelos salvos em:")
    print("  -", os.path.join(rtvc_root, "encoder", "saved_models"))
    print("  -", os.path.join(rtvc_root, "synthesizer", "saved_models"))
    print("  -", os.path.join(rtvc_root, "vocoder", "saved_models"))
