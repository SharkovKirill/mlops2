from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel

app = FastAPI()

class ArrayInput(BaseModel):
    data: list

@app.post("/prepare_data")
async def prepare_data(array_input: ArrayInput):
    try:
        np_array = np.array(array_input.data)
        
        result_array = np_array * 2
        
        return {"result": result_array.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)