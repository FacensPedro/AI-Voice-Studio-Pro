import os
import shutil
import tempfile
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from text_to_audio import text_to_speech
from audio_to_audio import audio_to_audio as process_audio_to_audio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_audio_outputs"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/text-to-audio/")
async def api_text_to_audio(
    text: str = Form(...),
    model_name: str = Form(...),
    speaker_idx: int = Form(0),
    speed: float = Form(1.0)
):
    try:
        unique_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(TEMP_DIR, unique_filename)

        text_to_speech(
            model_name=model_name,
            text=text,
            speaker_idx=speaker_idx,
            output_path=output_path,
            speed=speed
        )

        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Falha ao gerar o arquivo de áudio.")

        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=unique_filename,
            background_tasks=None # Para remover o arquivo após o envio, use BackgroundTasks
        )
    except FileNotFoundError as e:
        print(f"Erro de modelo não encontrado (verifique o model_name): {e}")
        raise HTTPException(status_code=400, detail=f"Modelo TTS não encontrado ou caminho inválido: {model_name}. Detalhe: {e}")
    except Exception as e:
        print(f"Erro em text_to_audio: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno: {str(e)}")

@app.post("/audio-to-audio/")
async def api_audio_to_audio(
    input_audio_file: UploadFile = File(...),
    tts_model: str = Form(...),
    speaker_idx: int = Form(0),
    speed: float = Form(1.0),
    whisper_model_size: str = Form("base")
):
    temp_input_path = None
    temp_output_path = None
    try:
        input_filename_unique = f"input_{uuid.uuid4()}_{input_audio_file.filename}"
        temp_input_path = os.path.join(TEMP_DIR, input_filename_unique)

        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(input_audio_file.file, buffer)

        if not os.path.exists(temp_input_path):
            raise HTTPException(status_code=500, detail="Falha ao salvar o arquivo de áudio de entrada.")

        output_filename_unique = f"output_{uuid.uuid4()}.wav"
        temp_output_path = os.path.join(TEMP_DIR, output_filename_unique)

        process_audio_to_audio(
            input_audio=temp_input_path,
            tts_model=tts_model,
            speaker_idx=speaker_idx,
            output_path=temp_output_path,
            speed=speed,
            whisper_model_size=whisper_model_size
        )

        if not os.path.exists(temp_output_path):
            raise HTTPException(status_code=500, detail="Falha ao processar e gerar o novo arquivo de áudio.")

        return FileResponse(
            path=temp_output_path,
            media_type="audio/wav",
            filename=output_filename_unique,
            background_tasks=None # Para remover os arquivos após o envio, use BackgroundTasks
        )
    except FileNotFoundError as e:
         print(f"Erro de modelo não encontrado (TTS ou Whisper): {e}")
         raise HTTPException(status_code=400, detail=f"Modelo TTS ou Whisper não encontrado ou caminho inválido. Detalhe: {e}")
    except Exception as e:
        print(f"Erro em audio_to_audio: {e}")
        raise HTTPException(status_code=500, detail=f"Ocorreu um erro interno no processamento áudio para áudio: {str(e)}")
    finally:
        # Limpeza básica dos arquivos temporários.
        # Para uma solução mais robusta, especialmente com BackgroundTasks,
        # a remoção seria feita após a resposta ser enviada.
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                # os.remove(temp_input_path) # Descomente se não usar BackgroundTasks para limpeza
                pass
            except Exception as e_clean:
                print(f"Erro ao limpar arquivo de entrada temporário {temp_input_path}: {e_clean}")
        
        # A FileResponse normalmente lida com o arquivo de saída se não for limpo por BackgroundTasks
        # Se precisar de limpeza explícita do output e não usar BackgroundTasks:
        # if temp_output_path and os.path.exists(temp_output_path) and not called_by_file_response:
        #     try:
        #          os.remove(temp_output_path)
        #     except Exception as e_clean:
        #          print(f"Erro ao limpar arquivo de saída temporário {temp_output_path}: {e_clean}")
        pass


if __name__ == "__main__":
    import uvicorn
    # Este uvicorn.run é para desenvolvimento local.
    # Para o Render, o 'Start Command' que você configurou (uvicorn main:app --host 0.0.0.0 --port $PORT) será usado.
    uvicorn.run(app, host="0.0.0.0", port=8000)
