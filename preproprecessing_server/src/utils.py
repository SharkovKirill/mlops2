from PIL import Image
import numpy as np
import requests
import json
from loguru import logger

def resize_to_gray_image(image_pil, target_size=(28, 28)):
    """
    Изменяет размер изображения.
    :param image: Изображение в формате numpy array.
    :param target_size: Целевой размер (ширина, высота).
    :return: Изображение с измененным размером.
    """
    gray_image = image_pil.convert("L")
    resized_image = np.array(gray_image.resize(target_size), dtype=np.float32)
    normalized_image = resized_image / 255
    return normalized_image.flatten().tolist()


def make_request_to_triton(image, model_name: str, model_version: int) -> list:
    triton_url = f"http://triton:8000/v2/models/{model_name}/versions/{model_version}/infer"
    data = {
        "inputs": [
            {
                "name": "input",
                "shape": [1, 784],
                "datatype": "FP32",
                "data": [image],
            }
        ]
    }
    response = requests.post(triton_url, data=json.dumps(data))
    response = response.json()
    logger.info(f"{response=}")
    return response['outputs'][0]['data']
