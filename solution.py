import pandas as pd
import numpy as np
from scipy.stats import norm


chat_id = 1112920502 # Ваш chat ID, не меняйте название переменной

def solution(x_success: int, 
             x_cnt: int, 
             y_success: int, 
             y_cnt: int) -> bool:
    x_conv = x_success / x_cnt
    y_conv = y_success / y_cnt
    
    pooled_conv = (x_success + y_success) / (x_cnt + y_cnt)
    se_pooled = np.sqrt(pooled_conv * (1 - pooled_conv) * (1/x_cnt + 1/y_cnt))
    z_score = (y_conv - x_conv) / se_pooled
    
    alpha = 0.03
    z_critical = norm.ppf(alpha/2)
    
    if z_score < -z_critical or z_score > z_critical:
        return True # Отключение рекламы выгодно
    else:
        return False # Отключение рекламы не выгодно
