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
import whisper # openai-whisper é importado como 'whisper'
from TTS.api import TTS

def transcribe_with_whisper(model_size, audio_path):
    """
    Usa o Whisper para transcrever o áudio de entrada em texto.
    Retorna: (texto_transcrito, idioma_detectado)
    """
    print(f"[LOG] transcribe_with_whisper: Iniciando. Model size: '{model_size}', Audio path: '{audio_path}'")
    
    print(f"[LOG] transcribe_with_whisper: Tentando carregar modelo Whisper '{model_size}'...")
    try:
        model = whisper.load_model(model_size)
        print(f"[LOG] transcribe_with_whisper: Modelo Whisper '{model_size}' carregado com sucesso.")
    except Exception as e:
        print(f"[LOG] transcribe_with_whisper: ERRO ao carregar modelo Whisper '{model_size}'. Detalhe: {e}")
        raise

    print(f"[LOG] transcribe_with_whisper: Transcrevendo áudio de '{audio_path}'...")
    try:
        result = model.transcribe(audio_path)
        texto = result["text"].strip()
        lang = result["language"]
        print(f"[LOG] transcribe_with_whisper: Transcrição: '{texto[:50]}...' (idioma: {lang})")
    except Exception as e:
        print(f"[LOG] transcribe_with_whisper: ERRO durante a transcrição com Whisper. Detalhe: {e}")
        raise
    
    print(f"[LOG] transcribe_with_whisper: Finalizado.")
    return texto, lang

def audio_to_audio(input_audio, tts_model, speaker_idx, output_path, speed, whisper_model_size):
    """
    1) Transcreve input_audio usando Whisper.
    2) Gera novo áudio TTS com o texto transcrito, usando modelo TTS (voz clonada).
    """
    print(f"[LOG] audio_to_audio: Iniciando. Input: '{input_audio}', TTS Model: '{tts_model}', Whisper Model: '{whisper_model_size}'")

    # 1) Transcrição
    print(f"[LOG] audio_to_audio: Chamando transcribe_with_whisper...")
    texto_transcrito, _ = transcribe_with_whisper(whisper_model_size, input_audio)
    print(f"[LOG] audio_to_audio: transcribe_with_whisper concluído.")

    # 2) Síntese TTS do texto transcrito
    print(f"[LOG] audio_to_audio: Tentando carregar modelo TTS '{tts_model}' (locutor idx={speaker_idx})...")
    try:
        tts_instance = TTS(tts_model)
        print(f"[LOG] audio_to_audio: Modelo TTS '{tts_model}' carregado com sucesso.")
    except Exception as e:
        print(f"[LOG] audio_to_audio: ERRO ao carregar modelo TTS '{tts_model}'. Detalhe: {e}")
        raise
        
    print(f"[LOG] audio_to_audio: Sintetizando texto transcrito em áudio: '{texto_transcrito[:50]}...'")
    try:
        wav = tts_instance.tts(texto_transcrito, speaker=speaker_idx, speed=speed)
        print(f"[LOG] audio_to_audio: Texto sintetizado com sucesso.")
    except Exception as e:
        print(f"[LOG] audio_to_audio: ERRO durante a síntese TTS. Detalhe: {e}")
        raise

    print(f"[LOG] audio_to_audio: Salvando áudio convertido em '{output_path}'...")
    try:
        tts_instance.save_wav(wav, output_path)
        print(f"[LOG] audio_to_audio: Áudio convertido salvo com sucesso em '{output_path}'.")
    except Exception as e:
        print(f"[LOG] audio_to_audio: ERRO ao salvar o arquivo WAV convertido. Detalhe: {e}")
        raise
    
    print(f"[LOG] audio_to_audio: Finalizado com sucesso.")


if __name__ == "__main__":
    print("[LOG] audio_to_audio.py: Executando como script principal.")
    parser = argparse.ArgumentParser(
        description="Converte Áudio->Áudio usando Whisper para transcrição e Coqui TTS para síntese."
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
        default="tiny", # Mantendo "tiny" como padrão aqui também para consistência
        help="Tamanho do modelo Whisper (tiny, base, small, medium, large)."
    )

    args = parser.parse_args()
    print(f"[LOG] audio_to_audio.py: Argumentos recebidos: {args}")
    
    if not os.path.isfile(args.input_audio):
        print(f"[LOG] audio_to_audio.py: ERRO - Arquivo de entrada '{args.input_audio}' não existe.")
        exit(1) # Usar sys.exit(1) seria mais canônico aqui

    audio_to_audio(
        input_audio=args.input_audio,
        tts_model=args.tts_model,
        speaker_idx=args.speaker_idx,
        output_path=args.output_path,
        speed=args.speed,
        whisper_model_size=args.whisper_model_size
    )
    print("[LOG] audio_to_audio.py: Script concluído.")
