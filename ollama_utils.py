import base64
import requests
from typing import Optional
import io
from PIL import Image
class OllamaClient:
    def __init__(self, base_url: str = "http://127.0.0.1:11434"):
        self.base_url = base_url
    def _encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    def analyze_image(self, image: Image.Image, prompt: Optional[str] = None) -> str:
        base64_image = self._encode_image(image)
        default_prompt = "Analyze the image and extract only the vehicle's license plate number. Provide no additional details or context."
        final_prompt = prompt if prompt else default_prompt
        payload = {
            "model": "gemma3:12b",
            "prompt": final_prompt,
            "images": [base64_image],
            "stream": False
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"Error communicating with Ollama: {str(e)}"
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"