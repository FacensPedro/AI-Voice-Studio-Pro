#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_to_audio.py

Converte um arquivo .mp3/.wav para um novo áudio no estilo da voz clonada, 
fazendo primeiro reconhecimento de fala com Whisper e depois síntese TTS.

Dependências:
    pip install whisper openai-whisper TTS numpy soundfile

Exemplo de uso:
    python audio_to_audio.py \
        --input_audio "entrada.mp3" \
        --tts_model "tts_models/pt/bracis/vits" \
        --speaker_idx 0 \
        --output_path "saida_clonada.wav"
"""

import argparse
import os
import whisper
from TTS.api import TTS

def transcribe_with_whisper(model_size, audio_path):
    """
    Usa o Whisper para transcrever o áudio de entrada em texto.
    Retorna: (texto_transcrito, idioma_detectado)
    """
    print(f">> Carregando modelo Whisper '{model_size}'...")
    model = whisper.load_model(model_size)
    print(">> Transcrevendo áudio...")
    result = model.transcribe(audio_path)
    texto = result["text"].strip()
    lang = result["language"]
    print(f">> Transcrição: {texto} (idioma: {lang})")
    return texto, lang

def audio_to_audio(input_audio, tts_model, speaker_idx, output_path, speed, whisper_model_size):
    """
    1) Transcreve input_audio usando Whisper.
    2) Gera novo áudio TTS com o texto transcrito, usando modelo TTS (voz clonada).
    """
    # 1) Transcrição
    texto, idioma = transcribe_with_whisper(whisper_model_size, input_audio)

    # 2) Síntese TTS do texto transcrito
    print(f">> Carregando modelo TTS '{tts_model}' (locutor idx={speaker_idx})...")
    tts = TTS(tts_model)
    print(">> Sintetizando texto transcrito em áudio...")
    wav = tts.tts(texto, speaker=speaker_idx, speed=speed)
    print(f">> Salvando áudio convertido em '{output_path}'...")
    tts.save_wav(wav, output_path)
    print(">> Conclusão: áudio convertido gerado com sucesso.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte Áudio→Áudio usando Whisper para transcrição e Coqui TTS para síntese."
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Caminho do arquivo de áudio de entrada (mp3 ou wav)."
    )
    parser.add_argument(
        "--tts_model",
        type=str,
        required=True,
        help="Modelo Coqui TTS para a síntese (p. ex. 'tts_models/pt/bracis/vits')."
    )
    parser.add_argument(
        "--speaker_idx",
        type=int,
        default=0,
        help="Índice do locutor no modelo TTS (se suportar múltiplos)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Arquivo WAV de saída com a voz clonada."
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Velocidade da síntese TTS (1.0 = normal)."
    )
    parser.add_argument(
        "--whisper_model_size",
        type=str,
        default="base",
        help="Tamanho do modelo Whisper (tiny, base, small, medium, large)."
    )

    args = parser.parse_args()
    if not os.path.isfile(args.input_audio):
        print(f"[ERRO] '{args.input_audio}' não existe.")
        exit(1)

    audio_to_audio(
        input_audio=args.input_audio,
        tts_model=args.tts_model,
        speaker_idx=args.speaker_idx,
        output_path=args.output_path,
        speed=args.speed,
        whisper_model_size=args.whisper_model_size
    )
