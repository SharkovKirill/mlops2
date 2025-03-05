from fastapi import FastAPI, HTTPException, UploadFile, File, Query
import numpy as np
from pydantic import BaseModel
import requests
import json
from utils import resize_to_gray_image, make_request_to_triton
from loguru import logger
from PIL import Image
import io

app = FastAPI()


class RequestParams(BaseModel):
    model_version: int


@app.post("/prepare_data")
async def prepare_data(
    model_version: int = Query(..., description="Версия модели"),
    model_name: str = Query(..., description="Название модели"),
    file: UploadFile = File(...),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = resize_to_gray_image(image)
        result = make_request_to_triton(image, model_name, model_version)
        logger.info(f"{result=}, {np.argmax(result)}")
        return {"result": str(np.argmax(result))}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8010)
