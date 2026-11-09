import numpy as np
import pandas as pd
from typing import Callable
from sklearn.metrics import mean_squared_error

def RMSE(
    y_data1: pd.Series,
    y_data2: pd.Series,
) -> float:
    y_data1_align, y_data2_align = y_data1.align(y_data2, join = "inner")
    RMSE = np.sqrt(mean_squared_error(y_data1_align, y_data2_align))
    return RMSE

def residual_derm(
    params: np.ndarray, 
    exp_data: pd.Series,
    f: Callable 
) -> pd.Series:
    sim_data = f(
        B = params[0], 
        S = params[1], 
        W = params[2],
        F = 0.0,
        M = 0.0,
        )
    exp_data_align, sim_data_align = sim_data.align(exp_data, join = "inner")
    residual_error = exp_data_align - sim_data_align
    return residual_error

def residual_sub(
    params: np.ndarray, 
    exp_data: pd.Series,
    f: Callable 
) -> pd.Series:
    sim_data = f(
        B = params[0], 
        S = params[1], 
        W = params[2], 
        F = params[3], 
        M = 0.0,
        )
    exp_data_align, sim_data_align = sim_data.align(exp_data, join = "inner")
    residual_error = exp_data_align - sim_data_align
    return residual_error

