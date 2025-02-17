# House_Price_Prediction
=======
 # Real Estate Price Prediction for Belgium

 ## 🏢 Description

 This project is focused on predicting real estate prices in Belgium using machine learning model. The data has been scraped from ImmoWeb, a real estate company in Belgium. The dataset is trained with XGBoost model and saved in the xgb2_model.pkl file. 

 ## 📊 Dataset
 The dataset used in this project contains information about real estate properties in Belgium, including details such as property type, price, provinces, living area, number of bedrooms, swimming pool, building condition, kitchen type,open fire and furnished. It  consists of 8356 houses.

 ## Deploying Method
 An API was created using FastAPI to make the trained model accessible for use by others. An interface was added to this API file using HTML and CSS, allowing users to input data and receive predicted prices. The API was deployed using a Docker file and Render. You can access the price prediction page at the following address

 https://house-price-prediction-b860.onrender.com


## Model Details
```
Used Model: XGBoost
Training score: 0.68
Testing score: 0.60
MSE: 15363835317.47

```

## 📦 Project structure
```
House_Price_Prediction
├── Dockerfile
├── README.md
├── data.csv
├── main.py
├── requirements.txt
├── scaler_X.pkl
├── templates
│   ├── base.html
│   └── prediction.html
├── train_final.py
└── xgb2_model.pkl

```

## ⏱️ Project Timeline
The initial setup of this project was completed in 7 days.
>>>>>>> 1a7f429 (correcting)