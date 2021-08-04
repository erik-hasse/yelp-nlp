from functools import lru_cache
import json
import random

from fastapi import FastAPI
from pydantic import BaseModel

from src.constants import data_dir


@lru_cache
def get_business_data():
    businesses = {}
    with open(data_dir / 'yelp_academic_dataset_business.json', 'rb') as f:
        for x in f:
            bus_data = json.loads(x)
            businesses[bus_data['business_id']] = {
                'name': bus_data['name'],
                'city': f'{bus_data["city"]}, {bus_data["state"]}'
            }
    return businesses


@lru_cache
def get_businesses():
    return list(get_business_data().keys())


def mock_retrieval_model(user_id):
    return random.sample(get_businesses(), 200)


def mock_ranking_model(user_id, candidates, city):
    bus_data = get_business_data()
    businesses = get_businesses()
    print(bus_data[businesses[0]])
    filtered_candidates = [
        bus for bus in businesses if bus_data[bus]['city'] == city
    ]
    return random.sample(filtered_candidates, min(len(filtered_candidates), 5))


class UserData(BaseModel):
    user_id: str
    city: str


class RestaurantData(BaseModel):
    name: str


app = FastAPI()


@app.get('/ping')
async def ping():
    return 'OK'


@app.post("/predict", response_model=list[RestaurantData])
async def predict(user: UserData):
    candidates = mock_retrieval_model(user.user_id)
    recommendations = mock_ranking_model(user.user_id, candidates, user.city)
    businesses = get_business_data()
    return [
        RestaurantData(name=businesses[name]['name'])
        for name in recommendations
    ]
