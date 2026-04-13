from fastapi import FastAPI, File, UploadFile, status, Response
from supabase import create_client, Client
from dotenv import load_dotenv
from PIL import Image

import numpy as np
import face_recognition
import io, os

# Configuracion Supabase ------------------------------------------------  
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# ------------------------------------------------
# Funciones Auxiliares ---------------------------
def extract_vector(content_bytes: bytes):
    # 1. Cargamos los bytes crudos a una imagen de Pillow
    image_pil = Image.open(io.BytesIO(content_bytes))

    # 2. Forzamos la imagen a RGB. Si es PNG con fondo transparente
    # lo vuelve negro/blanco
    image_rgb = image_pil.convert("RGB")

    # 3. face_recognition necesita un arreglo matematico (lista), no un objeto de imagen
    image_list = np.array(image_rgb)

    # Aquí ocurre la extracción de características
    encodings = face_recognition.face_encodings(image_list)

    if len(encodings) == 0:
        raise ValueError("No se detectaron rostros")
    if len(encodings) > 1:
        raise ValueError("Se detectaron varios rostros")
    
    return encodings[0].tolist()


app = FastAPI(title="Sistema de Reconocimiento Facial Universidad") 

@app.post("/reconocer")
async def recognize_face(response: Response, file: UploadFile = File(...)):
    try:
        # Leer la imagen que mandan en memoria
        content: bytes = await file.read()
        new_vector: list[float] = extract_vector(content)

        # 1. Extraer a todos los estudiantes
        db_response = supabase.table("Estudiantes").select("*").execute()
        students_db: list[dict] = db_response.data
        
        if not students_db:
            response.status_code = status.HTTP_404_NOT_FOUND
            return {
                "status": "error",
                "mensaje": "Estudiante no encontrado"
            }

        # 2. Comparar con cada estudiante
        for student in students_db:
            database_vector = np.array(student["vector_rostro"])

            # tolerance=0.5 es el nivel de exigencia. Mas bajo = mas estricto (Por defecto es 0.6)
            is_match: bool = face_recognition.compare_faces([database_vector], np.array(new_vector), tolerance=0.5)[0]
            
            if is_match:
                # 3. Registramos el ingreso
                supabase.table("Ingresos").insert({
                    "estudiante_id": student["id"],
                }).execute()

                response.status_code = status.HTTP_200_OK
                return {
                    "status": "success",
                    "mensaje": f"Acceso concedido. Bienvenido {student['nombres']} {student['apellidos']}",
                    "estudiante_id": student["id"],
                    "estudiante": f"{student['nombres']} {student['apellidos']}",
                }
            
        return {
            "status": "warning",
            "mensaje": "Rostro no reconocido! Intruso detectado",
        }
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "status": "fatal",
            "mensaje": f"Se rompió procesando la imagen: {str(e)}"
        }

# Registrar Estudiante (Iteración 2)
@app.post("/estudiantes")
async def register_student(response: Response, first_names: str, last_names: str, file: UploadFile = File(...)):
    try:  
        content = await file.read()
        vector = extract_vector(content)

        # Guardar en supabase
        new_student = {
            "nombres": first_names,
            "apellidos": last_names,
            "vector_rostro": vector
        }
        supabase.table("Estudiantes").insert(new_student).execute()

        response.status_code = status.HTTP_201_CREATED
        return {
            "status": "success",
            "mensaje": "Estudiante registrado correctamente"
        }
    except Exception as e:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {
            "status": "fatal",
            "mensaje": f"Se rompió procesando la imagen: {str(e)}"
        }
