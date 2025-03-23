# Imagen base con Python y OpenCV
FROM python:3.9

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de la aplicación
COPY app.py ollama_utils.py best.pt requirements.txt ./

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer puerto para FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
