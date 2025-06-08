
from datetime import date
import holidays
import pandas as pd

def calcula_coeficient_festiu(data: pd.Timestamp, json_data: dict) -> float:
    any_str = str(data.year)
    data_str = data.strftime("%Y-%m-%d")

    holidays = json_data["holidays"].get(any_str, [])
    comunitats = json_data["comunitats"]

    # Busquem si és festiu aquest dia
    festius_dia = [h for h in holidays if h["date"] == data_str]
    if not festius_dia:
        return 0.0  # No és festiu

    # Si algun dels festius és nacional, retornem 1.0
    for festiu in festius_dia:
        if festiu["counties"] is None:
            return 1.0

    # Si no és nacional, calculem quina proporció de població està de festa
    poblacio_total = sum(com["population_" + any_str] for com in comunitats.values())
    comunitats_en_festa = set()
    for festiu in festius_dia:
        comunitats_en_festa.update(festiu.get("counties", []))
    
    poblacio_festa = sum(
        comunitats[codi]["population_" + any_str] 
        for codi in comunitats_en_festa if codi in comunitats
    )

    coeficient = poblacio_festa / poblacio_total
    return round(coeficient, 4)



def get_type_of_day(date: pd.Timestamp) -> int:
    """
    Classifica un dia segons el seu tipus:
    0 = laborable
    1 = dissabte
    2 = diumenge
    3 = festiu estatals
    """
    espanya_festius = holidays.ES()

    if date in espanya_festius:
        return 3  # Festiu
    elif date.weekday() == 6:
        return 2  # Diumenge
    elif date.weekday() == 5:
        return 1  # Dissabte
    else:
        return 0  # Laborable
