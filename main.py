import asyncio

from fastapi import FastAPI
from langchain.text_splitter import MarkdownTextSplitter
from langchain.vectorstores.pgvector import PGVector,DistanceStrategy
from langchain.embeddings.base import  Embeddings
from langchain.schema import Document
from sse_starlette import EventSourceResponse
from langchain.docstore.document import Document as DDocument
import numpy as np
import multiprocessing as mp

mp.set_start_method("spawn")
manager = mp.Manager()

class EmbeddingsAdapter(Embeddings):
        pass
doc = Document()
app = FastAPI()
app.get()
np.linalg.norm()
norm = np.reshape()

asyncio.get_event_loop()
EventSourceResponse()

np.tile()
MarkdownTextSplitter()
pg = PGVector()
pg.delete_collection()
pg.similarity_search_with_score_by_vector()
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
