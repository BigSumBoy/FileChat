from sqlalchemy import create_engine
from sqlalchemy import text,Table, Column, Integer, String, MetaData
from sqlalchemy.orm import Session,relationship
from pydantic import BaseModel, Field

engine = create_engine()
metadata_obj = MetaData()
user_table = Table(
     "user_account",
     metadata_obj,
    Column("id", Integer, primary_key=True),
    Column("name", String(30)),
    Column("fullname", String),
 )
relationship()
with engine.connect() as conn:
    r = conn.execute()

with Session(engine) as session:
    result = session.execute().fetchall()

class User(BaseModel):
    id: Field(...)
    name: Field(...)