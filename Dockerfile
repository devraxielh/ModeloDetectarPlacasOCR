# Usa una imagen base con Python
FROM python:3.9

# Define el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia todos los archivos a la imagen del contenedor
COPY . /app

# Instala dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto en el que corre FastAPI
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
