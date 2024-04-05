from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from pydantic import Field
import httpx
import uvicorn

Field(..., alias="chat_openai")

client = httpx.Client(http2=True)
data = {'key': 'value'}
response = client.post('https://example.com/submit', json=data)
a = response.json()['models']
uvicorn.run()