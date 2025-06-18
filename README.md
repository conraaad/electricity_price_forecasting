
# Predicció de preus horaris del mercat elèctric (Espanya)

Aquest projecte forma part del **Treball de Fi de Grau en Enginyeria Informàtica** de la Facultat d'Informàtica de Barcelona (FIB) - Universitat Politècnica de Catalunya (UPC), i té com a objectiu desenvolupar un model predictiu capaç de **preveure el preu horari del mercat majorista de l’electricitat a Espanya**. L’objectiu principal és proporcionar una eina útil per a petites i mitjanes empreses industrials que volen planificar el seu consum de manera intel·ligent i eficient.

## 🧠 Enfocament

S’ha treballat amb tècniques de **machine learning supervisat**, centrant-se en models basats en arbres de decisió (Random Forest i XGBoost). El model s'entrena a partir de dades horàries del sistema elèctric espanyol i altres variables rellevants, utilitzant validació creuada temporal (*walk-forward validation*) per simular condicions reals de predicció.

## 📊 Dades i característiques

Les dades cobreixen el període **del 16 d’agost de 2021 al 15 de setembre de 2024**, amb un total de **27.032 observacions** i **41 variables**. Les fonts de dades inclouen:

- e.sios: Red Eléctrica (REE) – demanda, generació, intercanvis
- OMIE – preus del mercat elèctric
- AEMET – temperatura horària
- MIBGAS – preu del gas natural
- Nager.Date – Calendari de festius i codificacions temporals

Les **features seleccionades** finalment per a l'entrenament inclouen indicadors horaris, de demanda, de generació renovable, costos del gas, desequilibris i informació exògena com el calendari:

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

## ⚙️ Models i estratègies

S’han explorat diferents estratègies de modelització, amb i sense transformació logarítmica, i una estratègia *waterfall* que adapta el model al pas del temps:

|Model                      |	MAE (€/MWh) |	RMSE (€/MWh)	| SMAPE (%) |
|---------------------------|-------------|---------------|-----------|
|XGBoost Log Transform      |	10,8684     |	15,2534       |	61,1776   |
|XGBoost Waterfall          |	10,9118     |	14,5434       |	14,5434   |
|Random Forest Direct       |	9,6574      |	13,5126       |	58,1682   |
|Random Forest Log Transform|	8,9598      |	13,3900       |	56,6759   |

La millor estratègia general ha estat la combinació de **Random Forest amb transformació logarítmica** de la variable objectiu, que millora la predicció en trams de preus baixos i redueix la distorsió de les mètriques.

## 🧪 Validació
S’ha utilitzat `TimeSeriesSplit` de `scikit-learn` per aplicar validació creuada temporal, evitant filtració d’informació futura i simulant predicció real.

## 🛠️ Reproducció

### Requisits:
- Python 3.10+
- `scikit-learn`, `xgboost`, `pandas`, `numpy`, `matplotlib`, `joblib`

Per instal·lar totes les dependències d'aques projecte pot executar:

```bash
pip install -r requirements.txt
```

### Entrenament

Executa el fitxer que vulguis de `model/src/training/*.py` per entrenar i validar el model amb cross-validation.

### Predicció sobre test final

Executa el fitxer que vulguis de `model/src/predict_final/*.py` per obtenir els resultats sobre el conjunt de test separat.

## 🔌 Exposició via servei web

Un cop entrenat i validat el model, aquest s’ha integrat dins d’un servei web lleuger que permet fer prediccions a demanda mitjançant peticions HTTP. Aquest servei està pensat per ser consumit fàcilment des d’un client (frontend, app industrial o servei d’automatització).

L’API ha estat desenvolupada amb Django i exposa un endpoint `/predict` que retorna el preu horari estimat basat en dades existents, en aquest cas al ser una prova s'utlitzaran les dades del test.

Aquesta és la format de resposta de la petició `/predict` per al dia 2023-10-04 (dia del dataset de testing):

````json
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

### Per executar el projecte el servei en local:

1. Situar-se al directori `/service`.
2. Executar el següent:

    ````bash
    python manage.py makemigrations
    python manage.py migrate
    python manage.py runserver
    ````

## 💻 Interfície Web amb Flutter

Tot i que el desenvolupament del client no formava part dels objectius directes d’aquest Treball de Fi de Grau, s’ha implementat una interfície web amb Flutter Web amb l’objectiu de facilitar la presentació del projecte i mostrar el funcionament del servei en temps real.

Aquesta aplicació consumeix el servei REST exposat pel backend Django, i permet a qualsevol usuari registrat (a través del seu correu electrònic) obtenir les prediccions horàries del mercat elèctric d’un dia concret. També mostra mètriques d’error per hora (MAE, RMSE, SMAPE) i permet visualitzar les variables d’entrada utilitzades pel model per a cada predicció.

#### Característiques principals:
- 🌐 Desenvolupat amb Flutter Web (exclusivament per navegadors d’escriptori)
- 📩 Formulari de registre simple amb validació de correu
- 📈 Visualització del preu predit per hora amb gràfic interactiu
- 🧠 Detall de les features utilitzades en cada hora
- 🎯 Presentació de les mètriques d’error associades

**Nota:** Aquesta part queda fora de l’abast acadèmic de la memòria i no es recull amb detall en aquest document, però ha estat desenvolupada com a suport visual per a la defensa i com a eina pràctica per a la prova del sistema.

## 📁 Estructura del repositori


````
├── model/
│   ├── data/
│   │   ├── analysis/                # Datasets d'anàlisi
│   │   ├── datasets/                # Datasets definitius 
│   │   └── feature_data/            # Datasets de les features
│   └── src/                         # Codi
│       ├── data_treatment/          # Codi pel tractament de les dades
│       ├── predict_final/           # Models entrenats
│       └── training/                # Codi per l'entrenament dels models
├── service/                         # Servei REST
├── front/                           # Interfície web
├── .gitignore
├── LICENSE
├── README
└── requirements.txt
````

## 🧑🏼‍💻 Autor

Aquest projecte ha estat desenvolupat per en Conrad Puig i Arimon com a part del seu Treball de Fi de Grau en Enginyeria Informàtica.

## 📄 Llicència

Aquest projecte està llicenciat sota la **MIT License**, que permet l'ús, còpia, modificació i distribució lliure del codi, amb l'única condició que sempre que se'n citi l’autoria original.

Consulta el fitxer `LICENSE` per a més detalls.