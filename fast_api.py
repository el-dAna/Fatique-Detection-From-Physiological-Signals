
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from mylib.apifunctions import get_experiments

class User_input(BaseModel):
    project_name: str

app = FastAPI()

app.get("/")
def index():
    return {"name":"Ben"}

# app.post('/experiments')
# def get_experiment_list(input: User_input):
#     experiment_list = get_experiments(project_name=input.project_name)
#     return experiment_list


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9000)