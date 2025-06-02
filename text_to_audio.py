#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
text_to_audio.py

Geração de áudio a partir de texto (TTS) usando Coqui TTS.

Dependências:
    pip install TTS numpy librosa soundfile

Exemplo de uso:
    python text_to_audio.py \
        --model_name "tts_models/pt/bracis/vits" \
        --text "Olá, este é um teste de geração de áudio." \
        --speaker_idx 0 \
        --output_path "saida.wav"
"""

import argparse
import numpy as np
from TTS.api import TTS

def text_to_speech(model_name, text, speaker_idx, output_path, speed):
    """
    Usa Coqui TTS para gerar áudio de entrada textual.

    :param model_name: chave do modelo no hub do Coqui (ex: "tts_models/pt/bracis/vits")
    :param text: string a ser sintetizada
    :param speaker_idx: índice do locutor (caso o modelo suporte múltiplos locutores)
    :param output_path: local onde salvar o WAV gerado
    :param speed: velocidade de síntese (1.0 = normal, <1 mais lento, >1 mais rápido)
    """
    # Carrega o modelo TTS pré-treinado
    print(f">> Carregando modelo TTS '{model_name}'...")
    tts = TTS(model_name)

    # Gera numpy array com amostras (fs = 22050 Hz por padrão)
    print(">> Sintetizando texto...")
    wav = tts.tts(text, speaker=speaker_idx, speed=speed)

    # Salva em arquivo WAV
    tts.save_wav(wav, output_path)
    print(f">> Áudio salvo em '{output_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Geração de TTS usando Coqui TTS.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Nome do modelo Coqui TTS (ex: 'tts_models/pt/bracis/vits')"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Texto a ser sintetizado."
    )
    parser.add_argument(
        "--speaker_idx",
        type=int,
        default=0,
        help="Índice do locutor (se o modelo tiver múltiplos)."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Velocidade de síntese: 1.0 = normal."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Caminho do arquivo WAV de saída."
    )

    args = parser.parse_args()
    text_to_speech(
        model_name=args.model_name,
        text=args.text,
        speaker_idx=args.speaker_idx,
        output_path=args.output_path,
        speed=args.speed
    )
