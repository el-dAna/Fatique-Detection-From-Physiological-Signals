# from typing import Union
from fastapi import FastAPI

# from pydantic import BaseModel

# from mylib.apifunctions import get_experiments, delete_experiments

app = FastAPI()


# class Item(BaseModel):
#     name: str
#     price: float
#     is_offer: Union[bool, None] = None


# class User_train(BaseModel):
#     train_percent: float
#     sampling_window: int
#     degree_of_overlap: float
#     s3_model_name: str
#     clearml_task_name: str
#     epochs: int


@app.get("/")
def read_root():
    return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}


# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}


# @app.get("/exp/{project_name}/{task_name}")
# def get_experiment_list(project_name: str, task_name: str):
#     return get_experiments(project_name=project_name, task_name=task_name)


# @app.delete("/exp/{project_name}/")
# def delete_experiment_list(
#     task_name: str,
#     experiments_to_delete: list,
#     project_name=str,
# ):
#     return delete_experiments(
#         task_name="user delete", experiments_to_delete=["test_ui"]
#     )


# @app.post("/explist")
# def train_model(input: User_train):
#     #result = get_experiments(project_name=input.project_name, task_name=input.task_name)
#     return result
