import requests

# URL del servidor FastAPI
url = "http://127.0.0.1:8000/process_video/"

# Ruta del video que deseas enviar
video_path = r"C:\Users\Raxie\Downloads\PRUEBAS VALLEDUPAR - VLC\PRUEBAS VALLEDUPAR - VLC\LE3400_241022_084034_000872.ASF"

# Abrir el archivo y enviarlo en la solicitud
with open(video_path, "rb") as video_file:
    files = {"file": (video_path, video_file, "video/mp4")}
    response = requests.post(url, files=files)

# Imprimir la respuesta del servidor
print(response.status_code)
print(response.json())
