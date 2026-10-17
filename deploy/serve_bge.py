import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv
import os
load_dotenv()

model_path_or_name = os.getenv("BGE_M3_MODEL_PATH", "BAAI/bge-m3")
app = FastAPI(title="BGE-M3 Embedding API")

model = BGEM3FlagModel(
    model_path_or_name,
    use_fp16=torch.cuda.is_available(),
    devices=["cuda:0"] if torch.cuda.is_available() else ["cpu"]
)

class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=32)

@app.post("/embed")
def embed(request: EmbeddingRequest):
    try:
        output = model.encode(
            request.texts,
            batch_size=8,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        return {
            "model": "bge-m3",
            "dimension": 1024,
            "embeddings": output["dense_vecs"].tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))