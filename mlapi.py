# Bring in lightweight dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

class ScoringItem(BaseModel):
      isclosed: bool #1_
      dcoilwtico: float #50.0_
      oil_week_avg: float #55.0_
      weekofyear: float #30_
      isweekend: bool #0_
      startschool: float #1_
      year_2017: bool #1_
      quarter_2: bool #0_
      quarter_3: bool #1_
      quarter_4: bool #0_
      event_type_Black_F: bool #0_
      event_type_Bridge: bool #0_
      event_type_Cyber_M: bool #1_
      event_type_Dia_de: bool #0_
      event_type_Holiday: bool #0_
      event_type_Transfer: bool #0_
      event_type_Work_Day: bool #0_
      event_type_norm: bool #0_
      isevent_y: bool #1_
      trend: float #0.5_
      sin_1_freq_A_DEC: float #0.1_
      cos_1_freq_A_DEC: float #0.2_
      sin_2_freq_A_DEC: float #0.3_
      cos_2_freq_A_DEC: float #0.4_
      sin_3_freq_A_DEC: float #0.5_
      cos_3_freq_A_DEC: float #0.6_
      sin_4_freq_A_DEC: float #0.7_
      cos_4_freq_A_DEC: float #0.8_
      sin_5_freq_A_DEC: float #0.9_
      cos_5_freq_A_DEC: float #0.1_
      sin_1_freq_M: float #0.11_
      cos_1_freq_M: float #0.12_
      sin_2_freq_M: float #0.13_
      cos_2_freq_M: float #0.14_
      sin_1_freq_W_SUN: float #0.15_
      cos_1_freq_W_SUN: float #0.16_
      sin_2_freq_W_SUN: float #0.17_
      cos_2_freq_W_SUN: float #0.18_
      sin_3_freq_W_SUN: float #0.19_
      cos_3_freq_W_SUN: float #0.2

# loading models
with open("linear_regression_model.pkl","rb") as lr:
     linear_regression_model = pickle.load(lr)

with open("random_rorest_model.pkl","rb") as rf:
     random_rorest_model = pickle.load(rf)     

@app.post('/')
async def scoring_endpoint(item: ScoringItem):
    try:
        # Create a DataFrame with the input data
        df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
        
        # Make predictions using the loaded models
        res_1 = linear_regression_model.predict(df)
        res_2 = random_rorest_model.predict(df)

        yhat = abs((res_1 + res_2)/2)

        # Return the prediction as an integer
        return {"Predictions": yhat.tolist()}
    except Exception as e:
        # Handle exceptions gracefully (e.g., invalid input or model issues)
        return {"error": str(e)}