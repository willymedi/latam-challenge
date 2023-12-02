import pandas as pd
from datetime import datetime
import numpy as np
import xgboost as xgb
from typing import Tuple, Union, List

class DelayModel:

    def __init__(
        self
    ):
        self._model = None # Model should be saved in this attribute.
        self._use_top_10 = True
        self._columns = None

    def _get_min_diff(self, data):
        fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
        fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
        min_diff = ((fecha_o - fecha_i).total_seconds())/60
        return min_diff
    
    def _get_delay(self, data, target: str):
        data['min_diff'] = data.apply(self._get_min_diff, axis = 1)
        threshold_in_minutes = 15
        data[target] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)

    def _get_top_10_features(self, features):
                
        top_10_features = [
            "OPERA_Latin American Wings", 
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air"
        ]
        return features[top_10_features]

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        if target_column is not None:
            self._get_delay(data, target_column)
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix = 'OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix = 'TIPOVUELO'), 
            pd.get_dummies(data['MES'], prefix = 'MES')], 
            axis = 1
        )
        self._columns = features.columns.tolist()

        if self._use_top_10:
            features = self._get_top_10_features(features)

        if target_column is None:
            return features
        else:
            target = data[[target_column]]
            return features, target




        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
       

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        serie = target["delay"]
        n_y0 = len(serie[serie == 0])
        n_y1 = len(serie[serie == 1])
        scale = n_y0/n_y1
        print(f"Target: {target}")
        print(f"n_y0: {n_y0}, n_y1: {n_y1}")
        print(f"Scale: {scale}")
        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight = scale)
        self._model.fit(features, target)
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        return

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            return [0] * len(features[features.columns[0]])
        xgboost_y_preds = self._model.predict(features)
        xgboost_y_preds = [1 if y_pred > 0.5 else 0 for y_pred in xgboost_y_preds]
        return xgboost_y_preds

    def set_use_top_10(self, use_top_10: bool) -> None:
        self._use_top_10 = use_top_10
    
    def get_columns(self) -> List[str]:
        return self._columns