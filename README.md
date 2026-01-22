
# Hourly electricity market price prediction (Spain)

This project is part of the **Final Degree Project in Computer Engineering** of the Faculty of Computer Science of Barcelona (FIB) - Polytechnic University of Catalonia (UPC), and aims to develop a predictive model capable of **forecasting the hourly price of the wholesale electricity market in Spain**. The main objective is to provide a useful tool for small and medium-sized industrial companies that want to plan their consumption in an intelligent and efficient way.

## 🧠 Approach

We have worked with **supervised machine learning** techniques, focusing on models based on decision trees (Random Forest and XGBoost). The model is trained using hourly data from the Spanish electricity system and other relevant variables, using temporal cross-validation (*walk-forward validation*) to simulate real prediction conditions.

## 📊 Data and features

The data covers the period **from August 16, 2021 to September 15, 2024**, with a total of **27,032 observations** and **41 variables**. The data sources include:

- e.sios: Red Eléctrica (REE) – demand, generation, exchanges
- OMIE – electricity market prices
- AEMET – hourly temperature
- MIBGAS – natural gas price
- Nager.Date – Holiday calendar and time coding

The **features** finally selected for training include hourly indicators, demand, renewable generation, gas costs, imbalances and exogenous information such as the calendar:

```python
features = [
    'is_mond', 'is_tues','is_wed','is_thurs','is_fri','is_sat','is_sun',
    'is_sunday_or_holiday',
    'hour_sin', 'hour_cos',
    'type_day_workday','type_day_sat','type_day_sun','type_day_holiday',
    'holiday_coef', 'demand', 'low_demand',
    'solar_share_demand', 'wind_share_demand',
    'gas_generation_share', 'gas_price', 'residual_demand',
    'interchange_balance', 'renewable_ratio', 'high_renewable_ratio',
    'temp_dev', 'price_es_24h',
    'renewables_to_gas', 'demand_per_gas',
    'price_rolling_3h', 'gas_price_lag1'
]
```

## ⚙️ Models and strategies

Different modeling strategies have been explored, with and without logarithmic transformation, and a *waterfall* strategy that adapts the model over time:

|Model                      |	MAE (€/MWh) |	RMSE (€/MWh)  | SMAPE (%) |
|---------------------------|---------------|-----------------|-----------|
|XGBoost Log Transform      |	10,8684     |	15,2534       |	61,1776   |
|XGBoost Waterfall          |	10,9118     |	14,5434       |	14,5434   |
|Random Forest Direct       |	9,6574      |	13,5126       |	58,1682   |
|Random Forest Log Transform|	8,9598      |	13,3900       |	56,6759   |

The best overall strategy has been the combination of **Random Forest with logarithmic transformation** of the target variable, which improves prediction in low price ranges and reduces metric distortion.

## 🧪 Validation
`TimeSeriesSplit` from `scikit-learn` has been used to apply temporal cross-validation, avoiding future information leakage and simulating real prediction.

## 🛠️ Reproduction

### Requirements:
- Python 3.10+
- `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `joblib`, `django`

To install all the dependencies of this project you can run:

```bash
pip install -r requirements.txt
```

### Training

Run the file you want from `model/src/training/*.py` to train and validate the model with cross-validation.

### Prediction on final test

Run the file you want from `model/src/predict_final/*.py` to obtain the results on the separate test set.

## 🔌 Exposure via web service

Once the model has been trained and validated, it has been integrated into a lightweight web service that allows predictions to be made on demand via HTTP requests. This service is designed to be easily consumed from a client (frontend, industrial app or automation service).

The API has been developed with Django and exposes an endpoint `/predict` that returns the estimated hourly price based on existing data, in this case being a test the test data will be used.

This is the response format of the `/predict` request for the day 2023-10-04 (day of the testing dataset):

````
{
    'name_model': "random_forest_model",
    'date': "2023-10-04",
    'hour_predictions': {
        0 : {
            'predicted_price': float, 
            'mae': float, 
            'rmse': float,
            'smape': float
        },
        ...
        23 : {
            'predicted_price': float, 
            'mae': float, 
            'rmse': float,
            'smape': float
        },
    },
    'daily_mean': {
        'mae': float,
        'rmse': float,
        'smape': float
    }
}
````

### To run the service project locally:

1. Go to the `/service` directory.
2. Run the following:

    ````bash
    python manage.py makemigrations
    python manage.py migrate
    python manage.py runserver
    ````

## 💻 Web Interface with Flutter

Although client development was not part of the direct objectives of this Final Degree Project, a web interface with Flutter Web has been implemented with the aim of facilitating the presentation of the project and showing the operation of the service in real time.

This application consumes the REST service exposed by the Django backend, and allows any registered user (through their email) to obtain hourly predictions of the electricity market for a specific day. It also shows hourly error metrics (MAE, RMSE, SMAPE) and allows viewing the input variables used by the model for each prediction.

#### Main features:
- 🌐 Developed with Flutter Web (exclusively for desktop browsers)
- 📩 Simple registration form with email validation
- 📈 Visualization of the predicted price per hour with interactive graph
- 🧠 Detail of the features used in each hour
- 🎯 Presentation of the associated error metrics

**Note:** This part is outside the academic scope of the memory and is not covered in detail in this document, but has been developed as a visual support for the defense and as a practical tool for testing the system.

### Production execution of the frontend

To run the compiled web frontend locally, follow these steps:

1. Compile the optimized version of the frontend. This will generate the files in the build/web directory.
From the Flutter project root directory:
```
flutter build web
```

2. Serve the web application compiled with Python (port 8080)

```
cd build/web
python -m http.server 8080
```

Then open your browser and access:

```
http://localhost:8080
```
This method allows you to view the final version of the interface in a browser in a lightweight way, without the need for any additional server.


## 📁 Repository structure

````
├── model/
│   ├── data/
│   │   ├── analysis/                # Analysis Datasets
│   │   ├── datasets/                # Final Datasets 
│   │   └── feature_data/            # Features Datasets
│   └── src/                         # Code
│       ├── data_treatment/          # Data treatment code
│       ├── predict_final/           # Trained models
│       └── training/                # Training models code
├── service/                         # REST Service code
├── front/                           # Web UI
├── .gitignore
├── LICENSE
├── README
└── requirements.txt
````

## 🧑🏼‍💻 Author

This project was developed by Conrad Puig i Arimon as part of his Bachelor's Thesis in Computer Engineering.

## 📄 License

This project is licensed under the **MIT License**, which allows the free use, copying, modification and distribution of the code, with the only condition that the original authorship is always cited.

See the `LICENSE` file for more details.
