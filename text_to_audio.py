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
# import numpy as np # numpy é uma dependência do TTS, não necessariamente usado diretamente aqui.
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
    print(f"[LOG] text_to_speech: Iniciando. Model: '{model_name}', Speaker: {speaker_idx}, Speed: {speed}, Output: '{output_path}'")
    
    # Carrega o modelo TTS pré-treinado
    print(f"[LOG] text_to_speech: Tentando carregar modelo TTS '{model_name}'...")
    try:
        tts_instance = TTS(model_name)
        print(f"[LOG] text_to_speech: Modelo TTS '{model_name}' carregado com sucesso.")
    except Exception as e:
        print(f"[LOG] text_to_speech: ERRO ao carregar modelo TTS '{model_name}'. Detalhe: {e}")
        raise  # Re-levanta a exceção para ser capturada pelo chamador (main.py)

    # Gera numpy array com amostras
    print(f"[LOG] text_to_speech: Sintetizando texto: '{text[:50]}...'")
    try:
        wav = tts_instance.tts(text, speaker=speaker_idx, speed=speed)
        print(f"[LOG] text_to_speech: Texto sintetizado com sucesso.")
    except Exception as e:
        print(f"[LOG] text_to_speech: ERRO durante a síntese TTS. Detalhe: {e}")
        raise

    # Salva em arquivo WAV
    print(f"[LOG] text_to_speech: Salvando áudio em '{output_path}'...")
    try:
        tts_instance.save_wav(wav, output_path)
        print(f"[LOG] text_to_speech: Áudio salvo com sucesso em '{output_path}'.")
    except Exception as e:
        print(f"[LOG] text_to_speech: ERRO ao salvar o arquivo WAV. Detalhe: {e}")
        raise
    
    print(f"[LOG] text_to_speech: Finalizado.")


if __name__ == "__main__":
    print("[LOG] text_to_audio.py: Executando como script principal.")
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
    print(f"[LOG] text_to_audio.py: Argumentos recebidos: {args}")
    
    text_to_speech(
        model_name=args.model_name,
        text=args.text,
        speaker_idx=args.speaker_idx,
        output_path=args.output_path,
        speed=args.speed
    )
    print("[LOG] text_to_audio.py: Script concluído.")
