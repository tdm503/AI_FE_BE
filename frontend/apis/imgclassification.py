# apis/imgclassification.py
import requests
import io
from PIL import Image

BACKEND_URL = "http://127.0.0.1:8000"

def img_classification(image_path: str):
    image = Image.open(image_path)
    img_name = image_path.split("/")[-1]
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    url = f"{BACKEND_URL}/classification/img"
    files = {'file': (img_name, img_byte_arr, 'image/jpeg')}
    headers = {'accept': 'application/json'}

    response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        json_results = response.json()
        return "Success", json_results
    else:
        return f"Error: API request failed with status code {response.status_code}.", None




