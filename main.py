import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

model = pickle.load(open('model.pickle', 'rb'))
scaler = pickle.load(open('scaler.pickle', 'rb'))
encoder = pickle.load(open('encoder.pickle', 'rb'))
medians = pd.read_csv('medians.csv')

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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
	
    def to_df(self):
        data = pd.DataFrame({'name':self.name, 'year': self.year,'selling_price':self.selling_price,
            'km_driven':self.km_driven, 'fuel':self.fuel,'seller_type':self.seller_type,'transmission':self.transmission,
            'owner':self.owner, 'mileage':self.mileage,'engine':self.engine,'max_power':self.max_power,'torque':self.torque,
			'seats':self.seats
		}, index=[0])
        return data
	
    def format(self, data):
        data['mileage'] = data['mileage'].str.extract(r'(\d*\.?\d*)')
        data['mileage'] = data['mileage'].apply(lambda x: float(x) if x != '' else None)
        data['engine'] = data['engine'].str.extract(r'(\d*\.?\d*)')
        data['engine'] = data['engine'].apply(lambda x: float(x) if x != '' else None)
        data['max_power'] = data['max_power'].str.extract(r'(\d*\.?\d*)')
        data['max_power'] = data['max_power'].apply(lambda x: float(x) if x != '' else None)

        data = data.drop(columns=['torque'])

        data['mileage'] = data['mileage'].fillna(medians.loc[0,'mileage'])
        data['engine'] = data['engine'].fillna(medians.loc[0,'engine'])
        data['max_power'] = data['max_power'].fillna(medians.loc[0,'max_power'])
        data['seats'] = data['seats'].fillna(medians.loc[0,'seats'])

        data['engine'] = data['engine'].apply(int)
        data['seats'] = data['seats'].apply(int)


        data_enc = encoder.transform(data[['fuel','seller_type','transmission','owner','seats']])
        
        data_cat = np.hstack([data_enc, data[['km_driven', 'mileage', 'engine', 'max_power', 'seats', 'year']]])
        data_scaled = scaler.transform(data_cat)

        return data_scaled
	

    def predict(self, data):
        y = model.predict(data)
        return y

class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
	df = item.to_df()
	df = item.format(df)
	pred = item.predict(df)
	return pred[0]


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    results = []
    for item in items:
        df = item.to_df()
        df = item.format(df)
        results.append(item.predict(df)[0])
    return results
