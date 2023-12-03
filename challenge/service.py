from challenge.model import DelayModel
import pandas as pd
import os
from enum import Enum
from fastapi import HTTPException

from sklearn.model_selection import train_test_split


class Fields(Enum):
    OPERA = "OPERA"
    TIPOVUELO = "TIPOVUELO"
    MES = "MES"


class ApiService:

    def __init__(self,):
        self.model = DelayModel()
        self._columns = None

    def _validate_column(self, column_name, value):
        if value not in self._columns:
            raise HTTPException(status_code=400, detail=f"El valor de {column_name} no es válido")

    def initialize_model(self):
        data = pd.read_csv(filepath_or_buffer="./data/data.csv")
        self.model.set_use_top_10(False)
        features, target = self.model.preprocess(
            data=data,
            target_column="delay"
        )
        x_train, _, y_train, _ = train_test_split(features, target, test_size = 0.33, random_state = 42)
        self.model.fit(x_train, y_train)
        self._columns = features.columns.tolist()

    def predict(self, flight: dict) -> dict:
        opera = f"{Fields.OPERA.value}_{flight[Fields.OPERA.value]}"
        tipovuelo = f"{Fields.TIPOVUELO.value}_{flight[Fields.TIPOVUELO.value]}"
        mes = f"{Fields.MES.value}_{flight[Fields.MES.value]}"
        
        for column_name, value in zip([Fields.OPERA, Fields.TIPOVUELO, Fields.MES], [opera, tipovuelo, mes]):
            self._validate_column(column_name.value, value)

        input_data = self._prepare_for_predict(opera, tipovuelo, mes)
        prediction_result = self.model.predict(input_data)
        return {"predict": prediction_result}

    def _prepare_for_predict(self, opera:str, tipovuelo:str, mes:str) -> pd.DataFrame:
        index_opera = self._columns.index(opera)
        index_tipovuelo = self._columns.index(tipovuelo)
        index_mes = self._columns.index(mes)
        x = [0] * len(self._columns)
        x[index_opera] = 1
        x[index_tipovuelo] = 1
        x[index_mes] = 1
        return pd.DataFrame([x], columns=self._columns)


        