from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import os
import requests
from ultralytics import YOLO
from fastapi.responses import JSONResponse
from pathlib import Path
from ollama_utils import OllamaClient
from PIL import Image

app = FastAPI()
model = YOLO("runs/detect/detectorplaca/weights/best.pt")
ollama_client = OllamaClient()

OUTPUT_FOLDER = "./output"
Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

@app.post("/process_video/")
async def process_video(file: UploadFile = File(...)):
    try:
        temp_video_path = os.path.join(OUTPUT_FOLDER, file.filename)
        with open(temp_video_path, "wb") as f:
            f.write(await file.read())
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return JSONResponse(status_code=400, content={"error": "No se pudo abrir el video."})
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(0.5 * total_frames)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        output_video_path = os.path.join(OUTPUT_FOLDER, "output.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
        frame_count = start_frame
        detections = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 == 0:
                results = model(frame, device=0, conf=0.70)
                for r in results:
                    frame = r.plot()
                    for i, box in enumerate(r.boxes.xyxy):
                        x1, y1, x2, y2 = map(int, box.tolist())
                        cropped = frame[y1:y2, x1:x2]
                        crop_path = os.path.join(OUTPUT_FOLDER, f"detection_{frame_count}_{i}.jpg")
                        cv2.imwrite(crop_path, cropped)
                        detections.append(crop_path)
            out.write(frame)
            frame_count += 1
        cap.release()
        out.release()
        best_plate, plate_text, ollama_responses = extract_plate_text(detections)
        os.remove(temp_video_path)
        os.remove(output_video_path)
        for file in os.listdir(OUTPUT_FOLDER):
            file_path = os.path.join(OUTPUT_FOLDER, file)
            if file_path.endswith(".jpg") and file_path not in detections:
                os.remove(file_path)
        return JSONResponse(content={
            "message": "Proceso completado",
            "ollama_responses": ollama_responses,
            "best_plate": best_plate,
            "plate_text": plate_text
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/process_image/")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "No se pudo cargar la imagen."})
        results = model(frame, device=0, conf=0.70)
        detections = []
        for r in results:
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = map(int, box.tolist())
                cropped = frame[y1:y2, x1:x2]
                crop_path = os.path.join(OUTPUT_FOLDER, f"detection_image_{i}.jpg")
                cv2.imwrite(crop_path, cropped)
                detections.append(crop_path)
        best_plate, plate_text, ollama_responses = extract_plate_text(detections)
        return JSONResponse(content={
            "message": "Imagen procesada",
            "ollama_responses": ollama_responses,
            "best_plate": best_plate,
            "plate_text": plate_text
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def extract_plate_text(detections):
    best_plate = None
    plate_text = ""
    ollama_responses = {}
    for image_path in detections:
        with Image.open(image_path) as img:
            try:
                response = ollama_client.analyze_image(img, "Detect if this is a vehicle license plate and return only the plate number in the format ABC123.")
                ollama_responses[image_path] = response
                if response and len(response) == 6 and response[:3].isalpha() and response[3:].isdigit():
                    best_plate = image_path
                    plate_text = response
                    break
            except requests.exceptions.Timeout:
                ollama_responses[image_path] = "Error: Timeout en la consulta a Ollama"
            except requests.exceptions.RequestException as e:
                ollama_responses[image_path] = f"Error en la conexión a Ollama: {str(e)}"
    return best_plate, plate_text, ollama_responses
