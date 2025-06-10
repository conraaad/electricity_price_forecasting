
import os
import requests
from datetime import datetime, timedelta

def descarregar_marginals(any):
    carpeta = f'../data/source/target_price/{any}'
    os.makedirs(carpeta, exist_ok=True)

    start_date = datetime(any, 1, 13)
    end_date = datetime(any, 12, 31)
    data = start_date

    while data <= end_date:
        data_str = data.strftime('%Y%m%d')
        url = f"https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_{data_str}.1"
        desti = os.path.join(carpeta, f"marginalpdbc_{data_str}.1")

        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(desti, 'wb') as f:
                    f.write(response.content)
                print(f"✔ Descarregat: {data_str}")
            else:
                print(f"✘ No trobat: {data_str} (HTTP {response.status_code})")
        except Exception as e:
            print(f"⚠ Error amb {data_str}: {e}")

        data += timedelta(days=1)

# Descarregar fitxers per 2023 i 2024
# descarregar_marginals(2023)
descarregar_marginals(2024)
