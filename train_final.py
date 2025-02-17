
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score
from pickle import load

# Data preprocessing function
def load_and_preprocess_data (file_path):
    df = pd.read_csv(file_path)
    
    # Drop rows with missing values in specific columns
    df.dropna(subset=['Building condition'], inplace=True)
    df.dropna(subset=['Living area m²'], inplace=True)
    
    # Drop unnecessary columns
    df.drop(labels=['Terrace surface m²', 'Garden area m²'], axis=1, inplace=True)
    
    # Filter the dataset
    df = df[df['Price'] < 1000000]
    
    # Function to determine the province by postal code
    def get_province(postal_code):
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
        elif 1000 <= postal_code <= 1299:
            return 'Brussels'
        else:
            return 'Unknown'

    # Add a new column 'Province' based on the postal code
    df['Province'] = df['Locality data'].apply(get_province)

    # Label encoding for Building condition
    le = LabelEncoder()
    df['Building condition'] = le.fit_transform(df['Building condition'])

    # One-hot encoding for Property Type and Province
    ohe = OneHotEncoder(sparse_output=False, drop='first').set_output(transform="pandas")
    ohe_transform = ohe.fit_transform(df[['Property type', 'Province']])
    df = pd.concat([df, ohe_transform], axis=1).drop(columns=['Property type', 'Province'])

    # Drop unnecessary columns
    df.drop(labels=['Property ID', 'Locality data', 'Unnamed: 0'], axis=1, inplace=True)

    # Identify features and target variable
    X = df.drop(columns=['Price']) 
    y = df['Price']

    return X, y

# Train the model
def train_model(X_train, y_train):
    # Create and train the model
    model = XGBRegressor(n_estimators=1000, max_depth=3, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    model.fit(X_train, y_train)
    return model

#  Evaluate the model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print("Training score:", train_score)
    print("Testing score:", test_score)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)
    print("CV mean score:", cv_scores.mean())

    # Make predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print("MSE:", mse)

def main(file_path):
    # Load and preprocess data
    X, y = load_and_preprocess_data(file_path)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Rescale numeric features
    scaler_X = StandardScaler().fit(X_train)  # Fit on training data
    X_train_scaled = scaler_X.transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)


    # Scaler'ı .pkl dosyasına kaydet
    with open('scaler_X.pkl', 'wb') as f:
        pickle.dump(scaler_X, f)

    # Train the model
    xgb2_model = train_model(X_train_scaled, y_train)

    # Evaluate the model
    evaluate_model(xgb2_model, X_train_scaled, y_train, X_test_scaled, y_test)

    # Save the model
    with open('xgb2_model.pkl', 'rb') as file:
        modelUploaded = pickle.load(file)

    print("Model successufuly loaded!")
    

    # Örnek test verisi
    sample_data = X_test_scaled[0].reshape(1, -1)

    # Tahmin edilen fiyat
    predicted_price = modelUploaded.predict(sample_data)[0]

    # Gerçek fiyat
    actual_price = y_test.iloc[0]  # `y_test` DataFrame ya da Series olabilir, bu yüzden `.iloc[0]` kullanıyoruz.

    # Sonuçları yazdırma
    print(f"Predicted Price: {predicted_price:.2f}")
    print(f"Actual Price: {actual_price:.2f}")


if __name__ == "__main__":
    main('data.csv')  # Verisetinin yolunu değiştir