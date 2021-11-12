
from fastapi import FastAPI,File,UploadFile,Form
from typing import Optional,List

app = FastAPI()

@app.post("/push_image")
async def push(image_id:Optional[str]=None):
    img_id = image_id
    return {"images":imgs}

import uvicorn
uvicorn.run(app,host="0.0.0.0",port=9000)