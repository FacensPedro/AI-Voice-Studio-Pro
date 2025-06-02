import os
import shutil
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Supondo que text_to_audio.py e audio_to_audio.py estão no mesmo diretório
print("[LOG] main.py: Importando text_to_audio e audio_to_audio...")
from text_to_audio import text_to_speech
from audio_to_audio import audio_to_audio as process_audio_to_audio
print("[LOG] main.py: Módulos de processamento de áudio importados.")

print("[LOG] main.py: Inicializando FastAPI app...")
app = FastAPI()
print("[LOG] main.py: FastAPI app inicializado.")

print("[LOG] main.py: Configurando CORSMiddleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, restrinja para o seu domínio frontend.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("[LOG] main.py: CORSMiddleware configurado.")

TEMP_DIR = "temp_audio_outputs"
print(f"[LOG] main.py: Verificando/criando diretório temporário: {TEMP_DIR}")
os.makedirs(TEMP_DIR, exist_ok=True)
print(f"[LOG] main.py: Diretório temporário {TEMP_DIR} pronto.")

@app.post("/text-to-audio/")
async def api_text_to_audio(
    text: str = Form(...),
    model_name: str = Form(...),
    speaker_idx: int = Form(0),
    speed: float = Form(1.0)
):
    request_id = uuid.uuid4()
    print(f"[LOG] Req ID {request_id}: /text-to-audio/ endpoint chamado com texto: '{text[:30]}...', model_name: {model_name}")
    try:
        unique_filename = f"tts_output_{request_id}.wav"
        output_path = os.path.join(TEMP_DIR, unique_filename)
        print(f"[LOG] Req ID {request_id}: Caminho de saída definido como: {output_path}")

        print(f"[LOG] Req ID {request_id}: Chamando text_to_speech...")
        text_to_speech(
            model_name=model_name,
            text=text,
            speaker_idx=speaker_idx,
            output_path=output_path,
            speed=speed
        )
        print(f"[LOG] Req ID {request_id}: text_to_speech concluído.")

        if not os.path.exists(output_path):
            print(f"[LOG] Req ID {request_id}: ERRO - Arquivo de saída não encontrado em {output_path} após text_to_speech.")
            raise HTTPException(status_code=500, detail="Falha ao gerar o arquivo de áudio a partir do texto.")
        
        print(f"[LOG] Req ID {request_id}: Retornando arquivo: {output_path}")
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=unique_filename
        )
    except FileNotFoundError as e:
        print(f"[LOG] Req ID {request_id}: ERRO - Modelo TTS não encontrado: {model_name}. Detalhe: {e}")
        raise HTTPException(status_code=400, detail=f"Modelo TTS '{model_name}' não encontrado ou caminho inválido. Verifique o nome do modelo. Detalhe: {e}")
    except Exception as e:
        print(f"[LOG] Req ID {request_id}: ERRO - Exceção no endpoint /text-to-audio/: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno ao processar a solicitação de texto para áudio: {str(e)}")

@app.post("/audio-to-audio/")
async def api_audio_to_audio(
    input_audio_file: UploadFile = File(...),
    tts_model: str = Form(...),
    speaker_idx: int = Form(0),
    speed: float = Form(1.0),
    whisper_model_size: str = Form("tiny")
):
    request_id = uuid.uuid4()
    print(f"[LOG] Req ID {request_id}: /audio-to-audio/ endpoint chamado. TTS Model: {tts_model}, Whisper Model: {whisper_model_size}")
    temp_input_path = None
    temp_output_path = None
    try:
        input_suffix = os.path.splitext(input_audio_file.filename)[1]
        input_filename_unique = f"input_{request_id}{input_suffix}"
        temp_input_path = os.path.join(TEMP_DIR, input_filename_unique)
        print(f"[LOG] Req ID {request_id}: Salvando arquivo de entrada em: {temp_input_path}")

        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(input_audio_file.file, buffer)
        print(f"[LOG] Req ID {request_id}: Arquivo de entrada salvo.")

        if not os.path.exists(temp_input_path):
            print(f"[LOG] Req ID {request_id}: ERRO - Falha ao salvar o arquivo de áudio de entrada em {temp_input_path}.")
            raise HTTPException(status_code=500, detail="Falha ao salvar o arquivo de áudio de entrada.")

        output_filename_unique = f"a2a_output_{request_id}.wav"
        temp_output_path = os.path.join(TEMP_DIR, output_filename_unique)
        print(f"[LOG] Req ID {request_id}: Caminho de saída definido como: {temp_output_path}")

        print(f"[LOG] Req ID {request_id}: Chamando process_audio_to_audio...")
        process_audio_to_audio(
            input_audio=temp_input_path,
            tts_model=tts_model,
            speaker_idx=speaker_idx,
            output_path=temp_output_path,
            speed=speed,
            whisper_model_size=whisper_model_size
        )
        print(f"[LOG] Req ID {request_id}: process_audio_to_audio concluído.")

        if not os.path.exists(temp_output_path):
            print(f"[LOG] Req ID {request_id}: ERRO - Arquivo de saída não encontrado em {temp_output_path} após process_audio_to_audio.")
            raise HTTPException(status_code=500, detail="Falha ao processar e gerar o novo arquivo de áudio.")
        
        print(f"[LOG] Req ID {request_id}: Retornando arquivo: {temp_output_path}")
        return FileResponse(
            path=temp_output_path,
            media_type="audio/wav",
            filename=output_filename_unique
        )
    except FileNotFoundError as e:
        print(f"[LOG] Req ID {request_id}: ERRO - Modelo TTS ou Whisper não encontrado. TTS: {tts_model}, Whisper: {whisper_model_size}. Detalhe: {e}")
        raise HTTPException(status_code=400, detail=f"Modelo TTS '{tts_model}' ou Whisper (tamanho '{whisper_model_size}') não encontrado ou caminho inválido. Verifique os nomes. Detalhe: {e}")
    except Exception as e:
        print(f"[LOG] Req ID {request_id}: ERRO - Exceção no endpoint /audio-to-audio/: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno ao processar a solicitação de áudio para áudio: {str(e)}")
    finally:
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                # print(f"[LOG] Req ID {request_id}: Removendo arquivo de entrada temporário: {temp_input_path}")
                # os.remove(temp_input_path) # Considere remover via BackgroundTasks para não bloquear
                pass
            except Exception as e_clean_input:
                print(f"[LOG] Req ID {request_id}: AVISO - Erro ao tentar limpar o arquivo de entrada temporário {temp_input_path}: {e_clean_input}")
        # A limpeza do arquivo de saída (temp_output_path) é geralmente tratada pelo FileResponse,
        # ou pode ser explicitamente gerenciada com BackgroundTasks se necessário.
        pass

# Este bloco if __name__ == "__main__": é principalmente para desenvolvimento local.
# O Render usará o "Start Command" (ex: uvicorn main:app --host 0.0.0.0 --port $PORT)
if __name__ == "__main__":
    import uvicorn
    print("[LOG] main.py: Executando em modo de desenvolvimento local (uvicorn.run)...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
else:
    # Este print aparecerá nos logs do Render quando o Uvicorn importar o 'app'
    print("[LOG] main.py: Módulo carregado por um servidor ASGI (como Uvicorn no Render). 'app' está disponível.")
