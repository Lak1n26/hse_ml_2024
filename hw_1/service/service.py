import uvicorn
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import data_transform
from io import StringIO, BytesIO
app = FastAPI(title='Car price forecasting')


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float
     # selling_price: int  # убрал, посколько изначально мы не должны знать цену, а только прогнозируем ее


class Items(BaseModel):
    objects: List[Item]


@app.on_event("startup")
def fit_model():
    with open('models/model.pickle', "rb") as file:
        model = pickle.load(file)
    app.model = model
    with open('models/scaler.pickle', "rb") as file:
        scaler = pickle.load(file)
    app.scaler = scaler
    return


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame(item, columns=['feature', 'value'])  # переводим в датафрейм для дальнейшей обработки
    df.index = df.feature
    df = df[['value']].T.reset_index(drop=True)
    df = data_transform.get_final_data(df, app.scaler)
    predicted_price = app.model.predict(df)[0]
    return predicted_price


@app.post("/predict_items")
def predict_items(items=Body()):
    items_str = items.decode('utf-8')
    data = pd.read_csv(StringIO(items_str))
    prices = []
    for _, row in data.iterrows():
        df = data_transform.get_final_data(pd.DataFrame(row).T, app.scaler)
        predicted_price = app.model.predict(df)[0]
        prices.append(predicted_price)
        print(predicted_price)
    data['selling_price'] = prices
    output = BytesIO()
    data.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=output"
    })



if __name__ == "__main__":
    uvicorn.run("service:app", host="127.0.0.1", port=8000, reload=True)