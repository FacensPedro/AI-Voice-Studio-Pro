import os
import shutil
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Supondo que text_to_audio.py e audio_to_audio.py estão no mesmo diretório
from text_to_audio import text_to_speech
from audio_to_audio import audio_to_audio as process_audio_to_audio

app = FastAPI()

# Configuração do CORS para permitir requisições do seu frontend
# Idealmente, restrinja allow_origins para o domínio do seu frontend no Hugging Face Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ou especifique seu domínio do Hugging Face: ["https://SEU-USUARIO-HF-SEU-SPACE.hf.space"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_audio_outputs"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/text-to-audio/")
async def api_text_to_audio(
    text: str = Form(...),
    model_name: str = Form(...), # Ex: "tts_models/pt/bracis/vits"
    speaker_idx: int = Form(0),
    speed: float = Form(1.0)
):
    try:
        unique_filename = f"tts_output_{uuid.uuid4()}.wav"
        output_path = os.path.join(TEMP_DIR, unique_filename)

        # Chamada à função do seu script text_to_audio.py
        text_to_speech(
            model_name=model_name,
            text=text,
            speaker_idx=speaker_idx,
            output_path=output_path,
            speed=speed
        )

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Falha ao gerar o arquivo de áudio a partir do texto.")

        # Retorna o arquivo de áudio gerado
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=unique_filename
            # Para limpeza automática de arquivos após o envio, considere usar BackgroundTasks
            # from fastapi import BackgroundTasks
            # async def cleanup(file_path: str): os.remove(file_path)
            # E em FileResponse: background=BackgroundTasks().add_task(cleanup, output_path)
        )
    except FileNotFoundError as e:
        print(f"Erro de modelo TTS não encontrado: {e}")
        raise HTTPException(status_code=400, detail=f"Modelo TTS '{model_name}' não encontrado ou caminho inválido. Verifique o nome do modelo. Detalhe: {e}")
    except Exception as e:
        print(f"Erro no endpoint /text-to-audio/: {e}")
        # Em produção, evite expor detalhes internos do erro diretamente ao cliente por segurança.
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno ao processar a solicitação de texto para áudio: {str(e)}")

@app.post("/audio-to-audio/")
async def api_audio_to_audio(
    input_audio_file: UploadFile = File(...),
    tts_model: str = Form(...), # Ex: "tts_models/pt/bracis/vits"
    speaker_idx: int = Form(0),
    speed: float = Form(1.0),
    whisper_model_size: str = Form("tiny") # ALTERADO: Padrão para "tiny" para economizar memória
):
    temp_input_path = None
    temp_output_path = None
    try:
        # Salva o arquivo de áudio de entrada temporariamente
        input_suffix = os.path.splitext(input_audio_file.filename)[1]
        input_filename_unique = f"input_{uuid.uuid4()}{input_suffix}"
        temp_input_path = os.path.join(TEMP_DIR, input_filename_unique)

        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(input_audio_file.file, buffer)

        if not os.path.exists(temp_input_path):
             raise HTTPException(status_code=500, detail="Falha ao salvar o arquivo de áudio de entrada.")

        # Define o caminho para o arquivo de áudio de saída
        output_filename_unique = f"a2a_output_{uuid.uuid4()}.wav"
        temp_output_path = os.path.join(TEMP_DIR, output_filename_unique)

        # Chamada à função do seu script audio_to_audio.py
        process_audio_to_audio(
            input_audio=temp_input_path,
            tts_model=tts_model,
            speaker_idx=speaker_idx,
            output_path=temp_output_path,
            speed=speed,
            whisper_model_size=whisper_model_size # Passando o tamanho do modelo Whisper
        )

        if not os.path.exists(temp_output_path):
            raise HTTPException(status_code=500, detail="Falha ao processar e gerar o novo arquivo de áudio.")

        # Retorna o novo arquivo de áudio gerado
        return FileResponse(
            path=temp_output_path,
            media_type="audio/wav",
            filename=output_filename_unique
            # Considere BackgroundTasks para limpeza aqui também
        )
    except FileNotFoundError as e:
        print(f"Erro de modelo TTS ou Whisper não encontrado: {e}")
        raise HTTPException(status_code=400, detail=f"Modelo TTS '{tts_model}' ou Whisper (tamanho '{whisper_model_size}') não encontrado ou caminho inválido. Verifique os nomes. Detalhe: {e}")
    except Exception as e:
        print(f"Erro no endpoint /audio-to-audio/: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno ao processar a solicitação de áudio para áudio: {str(e)}")
    finally:
        # Limpeza simples dos arquivos temporários.
        # Uma abordagem mais robusta usaria BackgroundTasks para garantir a limpeza após a resposta.
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception as e_clean_input:
                print(f"Aviso: Erro ao tentar limpar o arquivo de entrada temporário {temp_input_path}: {e_clean_input}")
        # O FileResponse geralmente lida com o arquivo que está sendo enviado.
        # Se a limpeza do temp_output_path for necessária aqui (ex: se FileResponse falhar antes de enviar),
        # adicione lógica similar à do temp_input_path.
        pass


if __name__ == "__main__":
    import uvicorn
    # Este uvicorn.run é para desenvolvimento local.
    # No Render, o 'Start Command' que você configurou (uvicorn main:app --host 0.0.0.0 --port $PORT) será usado.
    print("Iniciando servidor Uvicorn para desenvolvimento local em http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
