from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle

class PredictRequest(BaseModel):
    JKT: int
    JKM: int
    GRS: int
    LT: int
    LB: int

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://klasrumah.69dev.id",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Load the trained model. (Pickle file)
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

def categorize_output(output):
        if output == 0:
            
            return ['Murah', "Rentang harga dibawah 1 Milliar"]
        elif output == 1:
            return ['Sedang', "Rentang harga 1 - 3 Milliar"]
        else:
            return ['Mahal', "Rentang harga diatas 3 Milliar"]

@app.get("/")
def read_root():
    return {"message": "API Model is running!"}


@app.post("/predict")
def predict(request: PredictRequest):    
    input_features = [request.JKT, request.JKM, request.GRS, request.LT, request.LB]
    features = np.array(input_features).reshape(1, -1)
    prediction = model.predict(features)  
    output = round(prediction[0], 2)
    category = categorize_output(int(output))


    return {"category": category[0], "detail": category[1]}
   

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)