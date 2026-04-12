from fastapi import FastAPI, File, UploadFile, status, Response
from PIL import Image
import numpy as np

import face_recognition
import io

app = FastAPI(title="Sistema de Reconocimiento Facial Universidad")

@app.post("/reconocer")
async def reconocer_rostro(response: Response, file: UploadFile = File(...)):
    # Leer la imagen que mandan en memoria
    content = await file.read()

    try:
        # 1. Cargamos los bytes crudos a una imagen de Pillow
        image_pil = Image.open(io.BytesIO(content))

        # 2. Forzamos la imagen a RGB. Si es PNG con fondo transparente
        # lo vuelve negro/blanco
        image_rgb = image_pil.convert("RGB")

        # 3. face_recognition necesita un arreglo matematico (lista), no un objeto de imagen
        image_list = np.array(image_rgb)

        # Aquí ocurre la extracción de características
        encodings = face_recognition.face_encodings(image_list)

        if len(encodings) == 0:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "status": "error",
                "status_code": 400,
                "mensaje": "No se detectaron rostros"
            }
        
        if len(encodings) > 1:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return {
                "status": "error",
                "status_code": 400,
                "mensaje": "Se detectaron varios rostros"
            }

        # Lo pasamos a lista normal porque JSON no entiende arreglos de Numpy
        encoding_rostro = encodings[0].tolist()

        return {
            "status": "success",
            "status_code": 200,
            "mensaje": "Rostro detectado correctamente",
            "vector_length": len(encoding_rostro),
            "vector": encoding_rostro
        }
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "status": "fatal",
            "status_code": 500,
            "mensaje": f"Se rompió procesando la imagen: {str(e)}"
        }