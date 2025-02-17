from fastapi import FastAPI, Request
import pickle
import numpy as np
from fastapi.templating import Jinja2Templates
from enum import Enum

# Loading XGBoost Model
filename = "xgb2_model.pkl"
with open(filename, "rb") as file:
    xgb2Uploaded = pickle.load(file)

#xgbUploaded = pickle.load(open('xgb2_model.pkl', 'rb'))

# Loading Scaler
scaler_X = pickle.load(open('scaler_X.pkl', 'rb'))

templates = Jinja2Templates(directory="templates")

app = FastAPI()

# Enum definition for Property Type
class PropertyType(str, Enum):
    house = "House"
    apartment = "Apartment"



# Province mapping (Post code -> Province one-hot encoding)
province_mapping = {
    'Flemish Brabant': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Antwerp excluded
    'Limburg': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'West Flanders': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'East Flanders': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'Walloon Brabant': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'Liège': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'Namur': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'Hainaut': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'Luxembourg': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'Brussels': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}
# Function to return province from postal code
def get_province(postal_code):
    # Flanders
    if 1500 <= postal_code <= 1999 or 3000 <= postal_code <= 3499:
        return 'Flemish Brabant'
    elif 2000 <= postal_code <= 2999:
        return 'Antwerp'
    elif 3500 <= postal_code <= 3999:
        return 'Limburg'
    elif 8000 <= postal_code <= 8999:
        return 'West Flanders'
    elif 9000 <= postal_code <= 9999:
        return 'East Flanders'
    
    # Wallonia
    elif 1300 <= postal_code <= 1499:
        return 'Walloon Brabant'
    elif 4000 <= postal_code <= 4999:
        return 'Liège'
    elif 5000 <= postal_code <= 5999:
        return 'Namur'
    elif 6000 <= postal_code <= 6599 or 7000 <= postal_code <= 7999:
        return 'Hainaut'
    elif 6600 <= postal_code <= 6999:
        return 'Luxembourg'
    
    # Brussels
    elif 1000 <= postal_code <= 1299:
        return 'Brussels'
    else:
        return None

# FastAPI root endpoint
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

# Predict endpoint
@app.get("/predict/")
async def make_prediction(
    request: Request,
    number_of_bedrooms: float,
    living_area: float,
    equipped_kitchen: str, # yes or no
    furnished : str, # yes or no
    open_fire : str, # yes or no
    swimming_pool: str,  # # yes or no
    building_condition: int,
    property_type: str,  
    post_code: int  # Post code input'u alıyoruz
):
   # try:
        
        # Property Type Mapping (House -> 1, Apartment -> 0)
        property_type_value = 1 if property_type.lower() == "House" else 0
        
        # equipped kitchen mapping (Yes -> 1, No -> 0)
        equipped_kitchen_value = 1 if equipped_kitchen.lower() == "Yes" else 0

        # furnished mapping (Yes -> 1, No -> 0)
        furnished_value = 1 if furnished.lower() == "Yes" else 0

        # open fire mapping (Yes -> 1, No -> 0)
        open_fire_value = 1 if open_fire.lower() == "Yes" else 0

        # Swimming pool mapping (Yes -> 1, No -> 0)
        swimming_pool_value = 1 if swimming_pool.lower() == "Yes" else 0

        # One-Hot Encoding for Property Type (House -> [1, 0], Apartment -> [0, 1])
        #property_type_value = property_type_mapping[property_type.value]
        #print(property_type_value)

        # Province mapping (Post code -> one-hot encoding)
        province_values = province_mapping.get(get_province(post_code), [0] * 10)  # As default it is 0
        

        # Preparing new data (modelin beklediği format)
        test_data = np.array([
            number_of_bedrooms,
            living_area,
            equipped_kitchen_value,
            furnished_value,
            open_fire_value,
            swimming_pool_value,
            building_condition,
            property_type_value,
        ]  + province_values).reshape(1, -1)

        
        # Scaling the new data
        scaled_data = scaler_X.transform(test_data)

        # Making Prediction
        prediction = xgb2Uploaded.predict(scaled_data)
        predicted_price = prediction[0]
        predicted_price=f"{predicted_price:,.2f}"
        
        

        return templates.TemplateResponse("prediction.html", {"request": request, "predicted_price": predicted_price})

   # except Exception as e:
        # Return the error message in case of an error
       # return {"error": str(e)}

