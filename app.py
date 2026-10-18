import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# ... (Importa el resto de librerías de tu bloque 1 de Colab) ...
print("🚀 Iniciando configuración del entorno PRO...")

# 1. Librerías estándar de Python
import os
import re
import json
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any, Callable

# 3. Machine Learning clásico
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 4. Modelos de Series de Tiempo (Statsmodels)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.tsa.seasonal import STL

# 5. Herramientas matemáticas y estadísticas (Scipy)
from scipy import stats
from scipy.stats import norm
from scipy.signal import periodogram
from scipy.optimize import minimize

# 6. Importación segura de modelos avanzados de Machine Learning
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("⚠️ XGBoost no disponible. Se omitirá sin reemplazarlo por otro modelo.")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
    print("⚠️ LightGBM no disponible. Se omitirá sin reemplazarlo por otro modelo.")

# 7. Configuración de entorno visual y advertencias globales
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
np.random.seed(42)

# =========================================================
# CONFIGURACIÓN GLOBAL DEL ALGORITMO (AUTO_CONFIG)
# =========================================================
# Límites computacionales y umbrales estadísticos para la búsqueda automática.
# Se definen aquí para facilitar ajustes según la capacidad de cómputo disponible.

AUTO_CONFIG: Dict[str, Any] = {
    "random_state": 42,                  # Semilla para reproducibilidad de modelos y Monte Carlo
    "max_cv_splits": 3,                  # Número máximo de particiones para validación cruzada (rolling-origin)
    "max_seasonal_candidates": 2,        # Top de periodos estacionales (N) a evaluar simultáneamente
    "max_seasonal_period": 24,           # Límite máximo para buscar ciclos (ej. 24 meses)
    "smoothing_random_candidates": 6,    # Candidatos aleatorios para optimizar alpha, beta, gamma
    "smoothing_local_maxiter": 12,       # Iteraciones máximas para refinamiento local en modelos de Holt-Winters
    "max_arima_configs": 72,             # Tope de combinaciones (p,d,q)(P,D,Q) para evitar desbordamiento de memoria
    "max_ml_windows": 4,                 # Máximo de ventanas temporales a probar en modelos de Machine Learning
    "n_rep_monte_carlo": 1000,           # Número de réplicas para la simulación de intervalos de confianza
    "uncertainty_max_origins": 30,       # Límite de orígenes históricos para construir el perfil de error
    "mc_min_errors_per_horizon": 5,      # Evidencia mínima (errores) exigida por horizonte para Monte Carlo
    "mc_min_complete_trajectories": 5,   # Mínimo de trayectorias completas requeridas para bootstrap
    "trend_min_r2": 0.10,                # R² mínimo para considerar que una tendencia lineal es significativa
}

print("✅ Librerías cargadas y entorno configurado (PRO).")
warnings.filterwarnings("ignore")

# Configuración de la página web
st.set_page_config(page_title="Herramienta Pronósticos S&OP", layout="wide")

# =========================================================================
# 🔻🔻🔻 PEGA AQUÍ TUS FUNCIONES MATEMÁTICAS (Bloques 2 al 9 de Colab) 🔻🔻🔻
#=========================================================
#BLOQUE 2 — MÉTRICAS DE ERROR
#=========================================================
#Cálculo estandarizado de métricas para evaluar la precisión de los modelos de pronóstico.Convención de error utilizada: e_t = F_t - D_t#"""

import numpy as np
from typing import List, Dict, Union

def safe_mape(y_true: Union[List[float], np.ndarray],
              y_pred: Union[List[float], np.ndarray],
              eps: float = 1e-8) -> float:
    """
    Calcula el Error Porcentual Absoluto Medio (MAPE) de forma segura.

    Excluye del cálculo los valores reales cercanos a cero para evitar
    divisiones por cero o distorsiones asintóticas.

    Args:
        y_true (ArrayLike): Valores reales de la demanda.
        y_pred (ArrayLike): Valores pronosticados.
        eps (float): Umbral de tolerancia para considerar un valor como cero.

    Returns:
        float: MAPE en porcentaje. Retorna np.nan si no hay valores válidos.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.abs(y_true) > eps

    if not np.any(mask):
        return np.nan
    return 100.0 * float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def mae(y_true: Union[List[float], np.ndarray],
        y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Calcula el Error Absoluto Medio (MAE o MAD).

    Args:
        y_true (ArrayLike): Valores reales de la demanda.
        y_pred (ArrayLike): Valores pronosticados.

    Returns:
        float: Promedio de los errores absolutos.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))

def mse(y_true: Union[List[float], np.ndarray],
        y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Calcula el Error Cuadrático Medio (MSE).

    Args:
        y_true (ArrayLike): Valores reales de la demanda.
        y_pred (ArrayLike): Valores pronosticados.

    Returns:
        float: Promedio de los errores al cuadrado.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: Union[List[float], np.ndarray],
         y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Calcula la Raíz del Error Cuadrático Medio (RMSE).

    Args:
        y_true (ArrayLike): Valores reales de la demanda.
        y_pred (ArrayLike): Valores pronosticados.

    Returns:
        float: Raíz cuadrada del MSE.
    """
    return float(np.sqrt(mse(y_true, y_pred)))

def mase(y_train: Union[List[float], np.ndarray],
         y_true: Union[List[float], np.ndarray],
         y_pred: Union[List[float], np.ndarray],
         eps: float = 1e-8) -> float:
    """
    Calcula el Error Escalado Absoluto Medio (MASE).

    Compara el MAE del modelo propuesto contra el MAE de un modelo
    ingenuo (Naïve) de un paso evaluado en el conjunto de entrenamiento.

    Args:
        y_train (ArrayLike): Serie histórica de entrenamiento.
        y_true (ArrayLike): Valores reales de la demanda en el período de prueba.
        y_pred (ArrayLike): Valores pronosticados en el período de prueba.
        eps (float): Factor de seguridad para evitar división por cero.

    Returns:
        float: Valor de MASE. Un valor < 1 indica que el modelo es mejor que el Naïve.
    """
    y_train = np.array(y_train, dtype=float)
    diffs = np.abs(np.diff(y_train))
    denom = float(np.mean(diffs)) if len(diffs) > 0 else eps
    denom = max(denom, eps)
    return float(mae(y_true, y_pred) / denom)

def bias(y_true: Union[List[float], np.ndarray],
         y_pred: Union[List[float], np.ndarray]) -> float:
    """
    Calcula el Sesgo o Error Medio (ME).

    Utiliza la convención e_t = F_t - D_t. Un valor positivo indica
    sobreestimación, mientras que un valor negativo indica subestimación.

    Args:
        y_true (ArrayLike): Valores reales de la demanda (D_t).
        y_pred (ArrayLike): Valores pronosticados (F_t).

    Returns:
        float: Promedio aritmético de los errores.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.mean(y_pred - y_true))

def metricas_error(y_train: Union[List[float], np.ndarray],
                   y_true: Union[List[float], np.ndarray],
                   y_pred: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """
    Consolida todas las métricas de error en un único diccionario.

    Args:
        y_train (ArrayLike): Datos de entrenamiento.
        y_true (ArrayLike): Datos reales de validación/prueba.
        y_pred (ArrayLike): Datos pronosticados.

    Returns:
        Dict[str, float]: Diccionario con MAPE, MAE, MSE, RMSE, MASE y BIAS.
    """
    return {
        "MAPE": safe_mape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_train, y_true, y_pred),
        "BIAS": bias(y_true, y_pred),
    }

def elegir_metrica_primaria(data: Union[List[float], np.ndarray]) -> str:
    """
    Selecciona dinámicamente la métrica primaria para optimizar modelos.

    Si más del 10% de la serie contiene ceros, el MAPE se vuelve inestable
    matemáticamente, por lo que el sistema cambia automáticamente a MASE.

    Args:
        data (ArrayLike): Serie de tiempo completa.

    Returns:
        str: Nombre de la métrica a utilizar ("MAPE" o "MASE").
    """
    y = np.asarray(data, dtype=float)
    pct_ceros = float(np.mean(np.isclose(y, 0.0))) if len(y) else 0.0
    return "MASE" if pct_ceros > 0.10 else "MAPE"

print("📐 Métricas documentadas y tipadas: MAPE, MAE, MSE, RMSE, MASE y Bias.")

"""
=========================================================
BLOQUE 3 — MODELOS CLÁSICOS DE SERIES DE TIEMPO
=========================================================
Implementación de modelos estadísticos tradicionales.
Incluye promedios, detección automática de estacionalidad,
Suavización Exponencial (SES, Holt, Winters en todas sus variantes)
y modelos de la familia ARIMA/SARIMA.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Any
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.seasonal import STL
from scipy.signal import periodogram

# --- 3.1 Promedios ---

def promedio_ventana_final(data: Union[List[float], np.ndarray], horizon: int, window: int) -> List[float]:
    """
    Pronostica utilizando un promedio simple sobre los últimos 'n' datos[cite: 3].

    Args:
        data (ArrayLike): Serie histórica de demanda.
        horizon (int): Número de períodos a pronosticar (k).
        window (int): Tamaño de la ventana (n) para calcular el promedio final.

    Returns:
        List[float]: Pronóstico constante para el horizonte definido.
    """
    data = list(data)
    n = len(data)
    if n == 0:
        return [0.0] * horizon
    window = int(window)
    window = max(1, min(window, n))
    avg = float(np.mean(data[-window:]))
    return [avg] * horizon

def optimize_promedio_ventana_final(train: List[float], test: List[float],
                                    n_min: int = 1, n_max: int = 10) -> Dict[str, Any]:
    """
    Optimiza el tamaño de la ventana 'n' iterando sobre posibles valores
    y seleccionando el que minimice el error frente a un conjunto de prueba.

    Args:
        train (List[float]): Datos de entrenamiento.
        test (List[float]): Datos de validación.
        n_min (int): Tamaño mínimo de la ventana.
        n_max (int): Tamaño máximo de la ventana.

    Returns:
        Dict[str, Any]: Diccionario con la mejor ventana ('n'), su error ('mape')
                        y los pronósticos ('pred').
    """
    best = {'n': None, 'mape': np.inf, 'pred': None}
    if len(train) == 0 or len(test) == 0:
        return {'n': 1, 'mape': np.inf, 'pred': [0.0] * len(test)}
    actual_n_max = min(max(1, n_max), len(train))
    actual_n_min = min(max(1, n_min), actual_n_max)
    for n in range(actual_n_min, actual_n_max + 1):
        pred = promedio_ventana_final(train, len(test), n)
        m = safe_mape(test, pred)
        score = m if np.isfinite(m) else mase(train, test, pred)
        if score < best['mape']:
            best = {'n': n, 'mape': score, 'pred': pred}
    if best['n'] is None:
        pred = promedio_ventana_final(train, len(test), 1)
        best = {'n': 1, 'mape': safe_mape(test, pred), 'pred': pred}
    return best

# --- 3.2 Detección automática de estacionalidad ---

def detectar_estacionalidad_automatica(data: Union[List[float], np.ndarray],
                                       max_period: Optional[int] = None,
                                       top_k: Optional[int] = None) -> Dict[str, Any]:
    """
    Identifica empíricamente los ciclos estacionales (N) más fuertes en la serie.

    Combina análisis de autocorrelación (ACF), análisis espectral (periodograma),
    descomposición STL y un test predictivo Naïve para calificar posibles frecuencias.

    Args:
        data (ArrayLike): Serie histórica.
        max_period (int, opcional): Período máximo a evaluar.
        top_k (int, opcional): Número máximo de candidatos a retornar.

    Returns:
        Dict[str, Any]: Contiene los candidatos estacionales (list), la tabla
                        de scores (DataFrame) y la serie sin tendencia (ndarray).
    """
    y = np.asarray(data, dtype=float)
    n = len(y)
    if n < 8:
        return {"candidatos": [], "tabla": pd.DataFrame(), "detrended": y.copy()}

    t = np.arange(n, dtype=float)
    lr = LinearRegression().fit(t.reshape(-1, 1), y)
    detrended = y - lr.predict(t.reshape(-1, 1))

    max_period = int(max_period or AUTO_CONFIG["max_seasonal_period"])
    max_period = min(max_period, max(2, n // 2))
    top_k = int(top_k or AUTO_CONFIG["max_seasonal_candidates"])

    try:
        acf_vals = acf(detrended, nlags=max_period, fft=True)
    except Exception:
        acf_vals = np.zeros(max_period + 1)

    try:
        freqs, power = periodogram(detrended)
        valid_power = power[freqs > 0]
        max_power = float(np.max(valid_power)) if len(valid_power) else 0.0
    except Exception:
        freqs, power, max_power = np.array([]), np.array([]), 0.0

    rows = []
    for s in range(2, max_period + 1):
        if n < 2 * s:
            continue

        acf_score = max(0.0, float(acf_vals[s])) if s < len(acf_vals) else 0.0

        spectral_score = 0.0
        if max_power > 0 and len(freqs) > 1:
            target = 1.0 / s
            idx = int(np.argmin(np.abs(freqs - target)))
            nearest = float(freqs[idx])
            rel_freq_error = abs(nearest - target) / max(target, 1e-12)
            proximity = max(0.0, 1.0 - rel_freq_error / 0.05)
            spectral_score = max(0.0, min(1.0, float(power[idx] / max_power))) * proximity

        stl_strength = 0.0
        try:
            stl = STL(detrended, period=s, robust=True).fit()
            var_r = float(np.var(stl.resid))
            var_sr = float(np.var(stl.seasonal + stl.resid))
            if var_sr > 1e-12:
                stl_strength = max(0.0, min(1.0, 1.0 - var_r / var_sr))
        except Exception:
            pass

        seasonal_errors = np.abs(detrended[s:] - detrended[:-s])
        baseline_errors = np.abs(detrended[s:])
        seasonal_mae = float(np.mean(seasonal_errors)) if len(seasonal_errors) else np.inf
        baseline_mae = float(np.mean(baseline_errors)) if len(baseline_errors) else np.inf
        improvement = 0.0
        if np.isfinite(seasonal_mae) and np.isfinite(baseline_mae) and baseline_mae > 1e-12:
            improvement = max(0.0, min(1.0, 1.0 - seasonal_mae / baseline_mae))

        score = 0.30 * acf_score + 0.25 * spectral_score + 0.25 * stl_strength + 0.20 * improvement
        rows.append({
            "N": s,
            "ACF": acf_score,
            "Periodograma": spectral_score,
            "STL": stl_strength,
            "Mejora_Naive": improvement,
            "Score": score,
        })

    tabla = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    candidatos = []

    if not tabla.empty:
        for _, row in tabla.iterrows():
            s = int(row["N"])
            if float(row["Score"]) < 0.22:
                continue
            if candidatos:
                top_score = float(tabla.loc[tabla["N"] == candidatos[0], "Score"].iloc[0])
                if float(row["Score"]) < 0.60 * top_score:
                    continue
            if not (float(row["Mejora_Naive"]) >= 0.10 or float(row["ACF"]) >= 0.35):
                continue

            redundant = False
            for c in candidatos:
                c_score = float(tabla.loc[tabla["N"] == c, "Score"].iloc[0]) if not tabla.loc[tabla["N"] == c].empty else np.inf
                is_multiple = (s % c == 0 or c % s == 0)
                is_neighbor = abs(s - c) <= max(1, int(round(0.15 * c)))
                if (is_multiple or is_neighbor) and float(row["Score"]) <= c_score * 1.10:
                    redundant = True
                    break
            if redundant:
                continue
            candidatos.append(s)
            if len(candidatos) >= top_k:
                break

    return {"candidatos": candidatos, "tabla": tabla, "detrended": detrended}

def base_seasonal_naive(data: Union[List[float], np.ndarray], horizon: int, seasonal_period: int) -> List[float]:
    """
    Genera un pronóstico ingenuo repitiendo el valor observado N períodos atrás.
    Actúa como benchmark base para evaluar modelos estacionales complejos.

    Args:
        data (ArrayLike): Serie histórica.
        horizon (int): Número de períodos futuros.
        seasonal_period (int): Longitud del ciclo estacional (N).

    Returns:
        List[float]: Valores proyectados ingenuamente.
    """
    data = list(map(float, data))
    s = int(seasonal_period)
    if s < 2:
        raise ValueError("seasonal_period debe ser >= 2.")
    if len(data) < s:
        raise ValueError(f"Seasonal Naïve requiere al menos {s} observaciones y recibió {len(data)}.")

    history = data.copy()
    preds = []
    for _ in range(int(horizon)):
        pred = float(history[-s])
        preds.append(pred)
        history.append(pred)
    return preds

# --- 3.3 Suavización Exponencial Simple (SES) ---

def base_ses(data: Union[List[float], np.ndarray], horizon: int, alpha: Optional[float] = None) -> List[float]:
    """
    Pronóstico mediante Suavización Exponencial Simple.

    Adecuado para series estacionarias. Aplica un conjunto de ponderaciones
    decrecientes a todos los datos pasados[cite: 3].

    Args:
        data (ArrayLike): Serie histórica.
        horizon (int): Número de períodos futuros.
        alpha (float, opcional): Constante de suavización del nivel.

    Returns:
        List[float]: Valores proyectados.
    """
    data = list(data)
    if len(data) == 0:
        return [0.0] * horizon
    try:
        model = ExponentialSmoothing(data, trend=None, seasonal=None, initialization_method='estimated')
        if alpha is None:
            fit = model.fit(optimized=True)
        else:
            fit = model.fit(smoothing_level=float(alpha), optimized=False)
        fc = np.array(fit.forecast(horizon), dtype=float)
        if np.all(np.array(data) >= 0):
            fc = np.maximum(fc, 0.0)
        return fc.tolist()
    except Exception as e:
        print(f"⚠️ SES falló ({e}), fallback a promedio.")
        return promedio_ventana_final(data, horizon, 1)

# --- 3.4 Modelo de Holt (Doble) ---

def base_holt(data: Union[List[float], np.ndarray], horizon: int,
              alpha: Optional[float] = None, beta: Optional[float] = None,
              trend: str = 'add') -> List[float]:
    """
    Pronóstico mediante Suavización Exponencial Doble (Modelo de Holt).

    Calcula dinámicamente un nivel y una tendencia lineal[cite: 2].
    Soporta tendencias aditivas o multiplicativas. No incluye estacionalidad.

    Args:
        data (ArrayLike): Serie histórica.
        horizon (int): Períodos a proyectar.
        alpha (float, opcional): Suavización del nivel.
        beta (float, opcional): Suavización de la tendencia.
        trend (str): 'add' (aditiva) o 'mul' (multiplicativa).

    Returns:
        List[float]: Valores proyectados.
    """
    data = list(data)
    if len(data) < 2:
        return base_ses(data, horizon, alpha=alpha)

    if trend == 'mul' and np.any(np.asarray(data) <= 0):
        raise ValueError("La tendencia multiplicativa requiere datos estrictamente positivos.")

    try:
        model = ExponentialSmoothing(data, trend=trend, seasonal=None, damped_trend=False, initialization_method='estimated')

        if alpha is None or beta is None:
            fit = model.fit(optimized=True)
        else:
            fit = model.fit(smoothing_level=float(alpha), smoothing_trend=float(beta), optimized=False)

        fc = np.array(fit.forecast(horizon), dtype=float)
        if np.all(np.array(data) >= 0):
            fc = np.maximum(fc, 0.0)
        return fc.tolist()
    except Exception as e:
        print(f"⚠️ Holt ({trend}) falló ({e}), fallback a SES.")
        return base_ses(data, horizon, alpha=alpha)

# --- 3.5 Modelo de Holt-Winters (Triple) ---

def base_holt_winters(data: Union[List[float], np.ndarray], horizon: int, seasonal_periods: int = 12,
                      alpha: Optional[float] = None, beta: Optional[float] = None,
                      gamma: Optional[float] = None, phi: Optional[float] = None,
                      trend: str = 'add', seasonal: str = 'add', damped_trend: bool = False) -> List[float]:
    """
    Pronóstico mediante Suavización Exponencial Triple (Holt-Winters).

    Integra componentes de nivel, tendencia y estacionalidad[cite: 1].
    Soporta las combinaciones: Add-Add, Add-Mul, Mul-Add, Mul-Mul.

    Args:
        data (ArrayLike): Serie histórica.
        horizon (int): Períodos futuros.
        seasonal_periods (int): Longitud del ciclo estacional (N).
        alpha, beta, gamma (float, opcional): Constantes de suavización.
        phi (float, opcional): Factor de amortiguación de tendencia.
        trend (str): 'add' o 'mul'.
        seasonal (str): 'add' o 'mul'.
        damped_trend (bool): Activa la tendencia amortiguada.

    Returns:
        List[float]: Valores proyectados.
    """
    data = list(data)
    n = len(data)
    seasonal_periods = int(seasonal_periods)

    if seasonal_periods < 2 or n < 2 * seasonal_periods:
        return base_holt(data, horizon, alpha=alpha, beta=beta, trend=trend)

    if (trend == 'mul' or seasonal == 'mul') and np.any(np.asarray(data) <= 0):
        raise ValueError("Las componentes multiplicativas requieren datos estrictamente positivos.")

    try:
        model = ExponentialSmoothing(
            data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            damped_trend=damped_trend,
            initialization_method='estimated'
        )

        if alpha is None or beta is None or gamma is None:
            fit = model.fit(optimized=True, use_brute=True)
        else:
            kwargs = dict(
                smoothing_level=float(alpha),
                smoothing_trend=float(beta),
                smoothing_seasonal=float(gamma),
                optimized=False,
            )
            if damped_trend:
                kwargs["damping_trend"] = float(phi if phi is not None else 0.98)
            fit = model.fit(**kwargs)

        fc = np.array(fit.forecast(horizon), dtype=float)

        if np.all(np.array(data) >= 0):
            fc = np.maximum(fc, 0.0)
        return fc.tolist()

    except Exception as e:
        raise RuntimeError(f"Holt-Winters ({trend}-{seasonal}) falló: {e}")

# --- 3.6 ARIMA / SARIMA ---

def base_arima(data: Union[List[float], np.ndarray], horizon: int,
               order: Tuple[int, int, int] = (1, 1, 1),
               seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
               trend: str = 'n') -> List[float]:
    """
    Ejecuta un modelo de la familia Box-Jenkins (ARIMA o SARIMA).

    Args:
        data (ArrayLike): Serie histórica.
        horizon (int): Períodos a proyectar.
        order (Tuple): Órdenes no estacionales (p, d, q).
        seasonal_order (Tuple): Órdenes estacionales (P, D, Q, s).
        trend (str): Parámetro de tendencia determinística ('n', 'c', 't', 'ct').

    Returns:
        List[float]: Pronósticos generados.
    """
    data = list(data)
    if len(data) == 0:
        return [0.0] * horizon
    try:
        model = SARIMAX(
            data,
            order=tuple(order),
            seasonal_order=tuple(seasonal_order),
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
        fc = np.array(model.forecast(horizon), dtype=float)
        if np.all(np.array(data) >= 0):
            fc = np.maximum(fc, 0.0)
        return fc.tolist()
    except Exception as e:
        print(f"⚠️ ARIMA/SARIMA {order} {seasonal_order} falló ({e}), fallback a promedio.")
        return promedio_ventana_final(data, horizon, 1)

def optimize_arima_iterative(train: List[float], test: List[float]) -> Dict[str, Any]:
    """
    Función de compatibilidad: Evalúa iterativamente un set básico de órdenes ARIMA
    para elegir el de menor error (MAPE/MASE).
    """
    orders = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1), (2,1,1), (1,1,2), (2,1,2)]
    best = {'order': None, 'mape': np.inf, 'pred': None}
    for order in orders:
        try:
            pred = base_arima(train, len(test), order=order)
            m = safe_mape(test, pred)
            score = m if np.isfinite(m) else mase(train, test, pred)
            if score < best['mape']:
                best = {'order': order, 'mape': score, 'pred': pred}
        except Exception:
            continue
    if best['order'] is None:
        pred = promedio_ventana_final(train, len(test), 1)
        best = {'order': (0,0,0), 'mape': np.inf, 'pred': pred}
    return best

print("✅ Modelos clásicos documentados: PROMEDIO, SEASONAL_NAIVE, SES, HOLT, HW (Todas las variaciones) y ARIMA/SARIMA.")

# =========================================================
# BLOQUE 4 — MODELOS DE MACHINE LEARNING (CON ANTIOVERFITTING S&OP)
# =========================================================
import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Any
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def create_features(data: Union[List[float], np.ndarray],
                    window: int,
                    seasonal_periods: Optional[List[int]] = None,
                    scale_n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    data = np.array(data, dtype=float)
    n = len(data)
    seasonal_periods = sorted(set(int(s) for s in (seasonal_periods or []) if int(s) >= 2))
    max_lag = max([window] + seasonal_periods) if seasonal_periods else window

    X, y = [], []
    scale_n = max(1, int(scale_n or n))

    if n <= max_lag:
        extra = 2 + len(seasonal_periods) + 2 * len(seasonal_periods)
        return np.empty((0, window + extra)), np.array([])

    for i in range(max_lag, n):
        win = data[i-window:i]
        trend = (win[-1] - win[0]) / (window - 1) if window > 1 else 0.0
        pos = i / scale_n

        seasonal_lags = [data[i-s] for s in seasonal_periods]

        fourier = []
        for s in seasonal_periods:
            fourier.extend([np.sin(2 * np.pi * i / s), np.cos(2 * np.pi * i / s)])

        X.append(list(win) + [trend, pos] + seasonal_lags + fourier)
        y.append(data[i])

    return np.array(X, dtype=float), np.array(y, dtype=float)

def _crear_modelo_ml(model_type: str, model_params: Optional[Dict[str, Any]] = None) -> Any:
    p = dict(model_params or {})
    rs = AUTO_CONFIG["random_state"]

    if model_type == 'rf':
        # Ajuste S&OP: Árboles menos profundos para evitar overfitting en series de negocio cortas
        defaults = dict(n_estimators=100, max_depth=5, min_samples_split=4, random_state=rs, n_jobs=-1)
        defaults.update(p)
        return RandomForestRegressor(**defaults)

    if model_type == 'xgb':
        if xgb is None:
            raise ImportError("XGBoost no está disponible en el entorno.")
        # Ajuste S&OP: Regularización y control de profundidad (max_depth=3)
        defaults = dict(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.9, random_state=rs, verbosity=0, n_jobs=-1)
        defaults.update(p)
        return xgb.XGBRegressor(**defaults)

    if model_type == 'lgb':
        if lgb is None:
            raise ImportError("LightGBM no está disponible en el entorno.")
        # Ajuste S&OP: num_leaves bajo para series cortas
        defaults = dict(n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=15, random_state=rs, verbose=-1, n_jobs=-1)
        defaults.update(p)
        return lgb.LGBMRegressor(**defaults)

    raise ValueError(f"Tipo ML no reconocido: {model_type}")

def train_ml_model(data: Union[List[float], np.ndarray],
                   horizon: int,
                   model_type: str = 'rf',
                   window: int = 5,
                   model_params: Optional[Dict[str, Any]] = None,
                   seasonal_periods: Optional[List[int]] = None) -> List[float]:
    data = list(map(float, data))
    n = len(data)
    if n == 0:
        return [0.0] * horizon

    seasonal_periods = sorted(set(int(s) for s in (seasonal_periods or []) if int(s) >= 2))
    window = int(max(2, min(window, max(2, n - 2))))
    max_lag = max([window] + seasonal_periods) if seasonal_periods else window

    # 1. Extracción de tendencia global (Clave para que los árboles puedan proyectar el futuro)
    X_time = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.asarray(data, dtype=float)
    trend_model = LinearRegression().fit(X_time, y)
    trend_fit = trend_model.predict(X_time)
    residuals = y - trend_fit

    if n <= max_lag + 2:
        future_idx = np.arange(n, n + horizon, dtype=float).reshape(-1, 1)
        fc = trend_model.predict(future_idx)
        if np.all(y >= 0):
            fc = np.maximum(fc, 0.0)
        return fc.tolist()

    # 2. Creación de variables y entrenamiento
    X, target = create_features(residuals, window, seasonal_periods=seasonal_periods, scale_n=max(1, n-1))
    if X.shape[0] < 3:
        return promedio_ventana_final(data, horizon, 1)

    model = _crear_modelo_ml(model_type, model_params)
    model.fit(X, target)

    # 3. Predicción iterativa paso a paso (recursiva)
    history_res = residuals.tolist()
    preds = []

    for step in range(horizon):
        i = len(history_res)
        hist_arr = np.asarray(history_res[-window:], dtype=float)

        if len(hist_arr) < window:
            hist_arr = np.pad(hist_arr, (window - len(hist_arr), 0), mode='edge')

        local_trend = (hist_arr[-1] - hist_arr[0]) / (window - 1) if window > 1 else 0.0
        pos = i / max(1, n-1)

        seasonal_lags = [history_res[i-s] if i-s >= 0 else 0.0 for s in seasonal_periods]
        fourier = []
        for s in seasonal_periods:
            fourier.extend([np.sin(2 * np.pi * i / s), np.cos(2 * np.pi * i / s)])

        x_new = np.array(list(hist_arr) + [local_trend, pos] + seasonal_lags + fourier, dtype=float).reshape(1, -1)

        pred_res = float(model.predict(x_new)[0])
        future_trend = float(trend_model.predict(np.array([[n + step]], dtype=float))[0])

        pred = future_trend + pred_res
        if np.all(y >= 0):
            pred = max(0.0, pred)

        preds.append(pred)
        history_res.append(pred_res)

    return preds

print("🤖 Bloque 4: Modelos ML documentados y optimizados contra overfitting.")

# =========================================================
# BLOQUE 5 — EVALUACIÓN DE MODELOS (TORNEO DINÁMICO S&OP)
# =========================================================
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Tuple, Dict, Optional, Callable, Any

MODELOS_DISPONIBLES: Dict[str, str] = {
    'PROM_SIMPLE': 'Promedio Simple (Toda la historia)',
    'PROM_MOVIL': 'Promedio Móvil (Ventana optimizada)',
    'PROM_PONDERADO': 'Promedio Móvil Ponderado (Optimizado)',
    'SEASONAL_NAIVE': 'Seasonal Naïve (Benchmark estacional)',
    'SES': 'Suavización Exponencial Simple (SES)',
    'HOLT_ADD': 'Holt Doble (Tendencia Aditiva)',
    'HOLT_MUL': 'Holt Doble (Tendencia Multiplicativa)',
    'HW_ADD_ADD': 'Holt-Winters Triple (Tend. Add, Est. Add)',
    'HW_ADD_MUL': 'Holt-Winters Triple (Tend. Add, Est. Mul)',
    'HW_MUL_ADD': 'Holt-Winters Triple (Tend. Mul, Est. Add)',
    'HW_MUL_MUL': 'Holt-Winters Triple (Tend. Mul, Est. Mul)',
    'ARIMA': 'ARIMA / SARIMA',
    'RF': 'Random Forest',
    'XGBOOST': 'XGBoost',
    'LIGHTGBM': 'LightGBM'
}

def crear_splits_temporales(n: int, cv_horizon: int, max_splits: int = 3, min_train: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n < 8: return []
    h = max(1, int(cv_horizon))
    min_train = int(min_train or max(8, n // 2))
    min_train = min(min_train, n - 1)
    h = min(h, n - min_train)
    possible = max(1, (n - min_train) // h) if h > 0 else 1
    n_splits = min(int(max_splits), possible)

    first_test_start = n - n_splits * h
    splits = []
    for i in range(n_splits):
        test_start = first_test_start + i * h
        test_end = min(n, test_start + h)
        if test_start < min_train or test_end <= test_start: continue
        splits.append((np.arange(test_start), np.arange(test_start, test_end)))
    return splits

def _calcular_riesgo_winkler(cv_actuals, cv_predictions, cv_errors, alpha=0.20):
    if len(cv_errors) < 2: return 100.0, np.inf
    std_err = float(np.std(cv_errors, ddof=1))
    if std_err == 0: std_err = 1e-5

    z = 1.28155 # Cuantil para alpha 0.20
    picp_count = 0
    winkler_scores = []

    for dt, ft in zip(cv_actuals, cv_predictions):
        lt, ut = ft - z * std_err, ft + z * std_err
        delta = ut - lt
        if dt < lt: winkler_scores.append(delta + (2.0 / alpha) * (lt - dt))
        elif dt > ut: winkler_scores.append(delta + (2.0 / alpha) * (dt - ut))
        else:
            winkler_scores.append(delta)
            picp_count += 1

    picp = (picp_count / len(cv_actuals)) * 100.0
    winkler_raw = np.mean(winkler_scores)
    mean_demanda = np.mean(cv_actuals)
    winkler_pct = (winkler_raw / mean_demanda * 100.0) if mean_demanda > 0 else np.inf
    return float(picp), float(winkler_pct)

def _evaluar_predictor_cv(data: List[float], splits: List[Tuple[np.ndarray, np.ndarray]], predictor: Callable, metric_name: str) -> Optional[Dict[str, Any]]:
    y = np.asarray(data, dtype=float)
    fold_metrics, error_paths, all_actual, all_pred, all_idx = [], [], [], [], []
    last_train, last_test, last_pred = None, None, None
    metas = []

    for train_idx, test_idx in splits:
        train, test = y[train_idx].tolist(), y[test_idx].tolist()
        try:
            pred, meta = predictor(train, len(test))
            pred = np.asarray(pred, dtype=float)
            if len(pred) != len(test) or np.any(~np.isfinite(pred)): continue

            met = metricas_error(train, test, pred)
            score = met.get(metric_name, np.nan)
            if not np.isfinite(score): continue

            errors = pred - np.asarray(test, dtype=float)
            fold_metrics.append(met)
            error_paths.append(errors.tolist())
            all_actual.extend(test)
            all_pred.extend(pred.tolist())
            all_idx.extend(test_idx.tolist())
            metas.append(meta or {})
            last_train, last_test, last_pred = train, test, pred.tolist()
        except Exception: continue

    if len(fold_metrics) != len(splits): return None

    avg = {}
    for key in fold_metrics[0].keys():
        vals = [m[key] for m in fold_metrics if np.isfinite(m.get(key, np.nan))]
        avg[key] = float(np.mean(vals)) if vals else np.nan

    return {
        "score_cv": float(avg[metric_name]), **avg,
        "cv_actuals": all_actual, "cv_predictions": all_pred, "cv_indices": all_idx,
        "cv_errors": (np.asarray(all_pred) - np.asarray(all_actual)).tolist(),
        "cv_error_paths": error_paths, "last_train": last_train or [], "last_test": last_test or [],
        "predictions": last_pred or [], "fold_meta": metas, "n_folds": len(fold_metrics),
    }

def _mejor_configuracion(data: List[float], splits: List[Tuple[np.ndarray, np.ndarray]], metric_name: str, configs: List[Dict[str, Any]], predictor_factory: Callable) -> Optional[Dict[str, Any]]:
    best = None
    for cfg in configs:
        ev = _evaluar_predictor_cv(data, splits, predictor_factory(cfg), metric_name)
        if ev is None: continue
        cand = {"params": cfg, **ev}
        if best is None or cand["score_cv"] < best["score_cv"]: best = cand
    return best

def _optimizar_suavizacion_cv(familia: str, data: List[float], splits: List[Tuple[np.ndarray, np.ndarray]], metric_name: str, estructura: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    estructura = dict(estructura or {})
    if familia == 'SES': names, bounds = ['alpha'], [(0.001, 0.999)]
    elif familia == 'HOLT': names, bounds = ['alpha', 'beta'], [(0.001, 0.999), (0.001, 0.999)]
    else:
        names = ['alpha', 'beta', 'gamma'] + (['phi'] if estructura.get('damped_trend') else [])
        bounds = [(0.001, 0.999)] * 3 + ([(0.80, 0.995)] if estructura.get('damped_trend') else [])

    def predictor_from_vector(vec):
        p = dict(zip(names, map(float, vec)))
        def predictor(train, h):
            if familia == 'SES': fc = base_ses(train, h, alpha=p['alpha'])
            elif familia == 'HOLT': fc = base_holt(train, h, alpha=p['alpha'], beta=p['beta'], trend=estructura.get('trend', 'add'))
            else:
                fc = base_holt_winters(train, h, seasonal_periods=estructura['seasonal_periods'], alpha=p['alpha'], beta=p['beta'], gamma=p['gamma'],
                                       phi=p.get('phi'), trend=estructura['trend'], seasonal=estructura['seasonal'], damped_trend=estructura['damped_trend'])
            return fc, p
        return predictor

    rng = np.random.default_rng(42)
    seeds = [[float(np.clip(v, lo, hi)) for lo, hi in bounds] for v in [0.10, 0.30, 0.50, 0.70, 0.90]]
    seeds.extend([[float(rng.uniform(lo, hi)) for lo, hi in bounds] for _ in range(6)])

    best_vec, best_ev = None, None
    for vec in seeds:
        ev = _evaluar_predictor_cv(data, splits, predictor_from_vector(vec), metric_name)
        if ev is not None and (best_ev is None or ev['score_cv'] < best_ev['score_cv']):
            best_vec, best_ev = np.asarray(vec, dtype=float), ev

    if best_vec is None: return None

    def objective(vec):
        ev = _evaluar_predictor_cv(data, splits, predictor_from_vector(vec), metric_name)
        return float(ev['score_cv']) if ev is not None else 1e12

    try:
        opt = minimize(objective, best_vec, method='Powell', bounds=bounds, options={'maxiter': 12, 'disp': False})
        if opt.success and np.isfinite(opt.fun) and float(opt.fun) < best_ev['score_cv']:
            refined = _evaluar_predictor_cv(data, splits, predictor_from_vector(opt.x), metric_name)
            if refined is not None: best_vec, best_ev = np.asarray(opt.x, dtype=float), refined
    except Exception: pass

    params = dict(zip(names, map(float, best_vec)))
    params.update(estructura)
    return {"params": params, **best_ev}

def _optimizar_pmp_cv(data: List[float], splits: List[Tuple[np.ndarray, np.ndarray]], metric_name: str, max_window: int = 5) -> Optional[Dict[str, Any]]:
    best_ev, best_w, best_n = None, None, None
    for n in range(2, max_window + 1):
        if len(data) < n + 2: continue
        w0 = np.ones(n) / n
        bounds = [(0.0, 1.0)] * n
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}

        def predictor_from_weights(w, win=n):
            def predictor(train, h):
                train_data = list(train)
                w_norm = np.clip(w, 0, 1)
                w_norm = w_norm / w_norm.sum() if w_norm.sum() > 0 else np.ones(win) / win
                pred = float(np.mean(train_data)) if len(train_data) < win else float(np.dot(train_data[-win:], w_norm))
                return [pred] * h, {'window': win, 'weights': w_norm.tolist()}
            return predictor

        def objective(w):
            ev = _evaluar_predictor_cv(data, splits, predictor_from_weights(w), metric_name)
            return float(ev['score_cv']) if ev is not None else 1e12

        try:
            opt = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 50, 'disp': False})
            if opt.success and np.isfinite(opt.fun):
                refined_w = np.clip(opt.x, 0, 1)
                refined_w = refined_w / refined_w.sum() if refined_w.sum() > 0 else np.ones(n) / n
                ev = _evaluar_predictor_cv(data, splits, predictor_from_weights(refined_w), metric_name)
                if ev is not None and (best_ev is None or ev['score_cv'] < best_ev['score_cv']):
                    best_ev, best_w, best_n = ev, refined_w, n
        except Exception: continue

    if best_ev is not None:
        return {"params": {'window': best_n, 'weights': best_w.tolist()}, **best_ev}
    return None

def _generar_configs_arima(data: List[float], seasonal_periods: Optional[List[int]]) -> List[Dict[str, Any]]:
    configs = []
    orders = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]
    for o in orders:
        configs.append({'order': o, 'seasonal_order': (0,0,0,0), 'trend': 'n'})
        configs.append({'order': o, 'seasonal_order': (0,0,0,0), 'trend': 'c'})
    for s in (seasonal_periods or []):
        for o in [(1,0,0), (0,1,1)]:
            configs.append({'order': o, 'seasonal_order': (1,0,0,int(s)), 'trend': 'n'})
            configs.append({'order': o, 'seasonal_order': (0,1,0,int(s)), 'trend': 'n'})
    return configs

def _evaluar_individual(nombre: str, data: List[float], splits: List[Tuple[np.ndarray, np.ndarray]], seasonal_periods_hw: List[int], metric_name: str) -> Optional[Dict[str, Any]]:
    y_array = np.asarray(data)
    positivos = np.all(y_array > 0)
    best = None

    if nombre == 'PROM_SIMPLE':
        best = _mejor_configuracion(data, splits, metric_name, [{'window': 'all'}], lambda cfg: (lambda train, h: (promedio_ventana_final(train, h, len(train)), {'window': len(train)})))
        if best: best['modelo'] = "PROMEDIO_SIMPLE"
    elif nombre == 'PROM_MOVIL':
        best = _mejor_configuracion(data, splits, metric_name, [{'window': w} for w in range(1, max(1, min(24, len(data)//3)) + 1)], lambda cfg: (lambda train, h: (promedio_ventana_final(train, h, cfg['window']), cfg)))
        if best: best['modelo'] = f"PROMEDIO_MOVIL(n={best['params']['window']})"
    elif nombre == 'PROM_PONDERADO':
        best = _optimizar_pmp_cv(data, splits, metric_name, max_window=max(2, min(10, len(data)//3)))
        if best: best['modelo'] = f"PROM_PONDERADO(n={best['params']['window']})"
    elif nombre == 'SEASONAL_NAIVE':
        for s in list(seasonal_periods_hw or []):
            cfg = {'seasonal_period': int(s)}
            cand = _evaluar_predictor_cv(data, splits, lambda train, h, s=int(s): (base_seasonal_naive(train, h, s), {'seasonal_period': s}), metric_name)
            if cand is not None and (best is None or cand['score_cv'] < best['score_cv']): best = {'params': cfg, **cand}
        if best: best['modelo'] = f"SEASONAL_NAIVE(N={best['params']['seasonal_period']})"
    elif nombre == 'SES':
        best = _optimizar_suavizacion_cv('SES', data, splits, metric_name)
        if best: best['modelo'] = f"SES(alpha={best['params']['alpha']:.4f})"
    elif nombre == 'HOLT_ADD':
        best = _optimizar_suavizacion_cv('HOLT', data, splits, metric_name, {'trend': 'add'})
        if best: best['modelo'] = f"HOLT_ADD(alpha={best['params']['alpha']:.4f})"
    elif nombre == 'HOLT_MUL' and positivos:
        best = _optimizar_suavizacion_cv('HOLT', data, splits, metric_name, {'trend': 'mul'})
        if best: best['modelo'] = f"HOLT_MUL(alpha={best['params']['alpha']:.4f})"
    elif nombre.startswith('HW_'):
        parts = nombre.split('_')
        trend_type, seas_type = ('add' if parts[1] == 'ADD' else 'mul'), ('add' if parts[2] == 'ADD' else 'mul')
        if not ((trend_type == 'mul' or seas_type == 'mul') and not positivos):
            for s in list(seasonal_periods_hw or []):
                if len(data) < 2 * int(s) + 2: continue
                for dt in [False]:
                    cand = _optimizar_suavizacion_cv('HW', data, splits, metric_name, {'seasonal_periods': int(s), 'trend': trend_type, 'seasonal': seas_type, 'damped_trend': dt})
                    if cand is not None and (best is None or cand['score_cv'] < best['score_cv']): best = cand
            if best: best['modelo'] = f"HW_{trend_type.upper()}_{seas_type.upper()}(N={best['params']['seasonal_periods']})"
    elif nombre == 'ARIMA':
        best = _mejor_configuracion(data, splits, metric_name, _generar_configs_arima(data, seasonal_periods_hw), lambda cfg: (lambda train, h: (base_arima(train, h, cfg['order'], cfg['seasonal_order'], cfg['trend']), cfg)))
        if best: best['modelo'] = f"ARIMA{best['params']['order']}" if best['params']['seasonal_order'] == (0,0,0,0) else f"SARIMA{best['params']['order']}x{best['params']['seasonal_order']}"
    elif nombre in ['RF', 'XGBOOST', 'LIGHTGBM']:
        mt = {'RF': 'rf', 'XGBOOST': 'xgb', 'LIGHTGBM': 'lgb'}[nombre]
        if not (nombre == 'XGBOOST' and xgb is None) and not (nombre == 'LIGHTGBM' and lgb is None):
            best = _mejor_configuracion(data, splits, metric_name, [{'window': w, 'seasonal_periods': seasonal_periods_hw, 'model_params': {}} for w in ([3, 4, 5] if len(data) < 30 else [3, 5, 12])],
                lambda cfg: (lambda train, h: (train_ml_model(train, h, model_type=mt, window=cfg['window'], model_params=cfg['model_params'], seasonal_periods=cfg['seasonal_periods']), cfg)))
            if best: best['modelo'] = f"{nombre}(window={best['params']['window']})"

    if best is None: return None
    best['id'] = nombre
    best['metrica_principal'] = metric_name
    best.update({k.lower(): best[k] for k in ['MAPE', 'MAE', 'MSE', 'RMSE', 'MASE', 'BIAS']})

    picp, winkler_pct = _calcular_riesgo_winkler(best['cv_actuals'], best['cv_predictions'], best['cv_errors'])
    best['picp'] = picp
    best['winkler'] = winkler_pct
    return best

def _candidatos_estacionales_cv_sin_fuga(data: List[float], splits: List[Tuple[np.ndarray, np.ndarray]]):
    if not splits: return [], {"candidatos": [], "tabla": pd.DataFrame(), "detrended": np.asarray(data, dtype=float)}
    return list(detectar_estacionalidad_automatica(np.asarray(data, dtype=float)[splits[0][0]].tolist()).get('candidatos', [])), {}

def evaluar_modelos_seleccionados(data: List[float], seasonal_periods_hw: Optional[List[int]] = None, cv_horizon: Optional[int] = None, min_train: Optional[int] = None) -> List[Dict[str, Any]]:
    full_data = list(map(float, data))
    h = int(cv_horizon or max(1, min(12, len(full_data)//6)))
    splits = crear_splits_temporales(len(full_data), h, max_splits=3, min_train=min_train)
    if not splits: return []

    final_seasonal_periods = seasonal_periods_hw if seasonal_periods_hw else _candidatos_estacionales_cv_sin_fuga(full_data, splits)[0]

    # ¡AQUÍ LA MAGIA S&OP! SELECCIÓN DINÁMICA DE MÉTRICA (MAPE vs MASE)
    metrica_opt = elegir_metrica_primaria(full_data)

    print("\nEVALUACIÓN AUTOMÁTICA (TORNEO S&OP)")
    print("="*70)
    print(f"🎯 Métrica de optimización dinámica activada: {metrica_opt}\n")

    results = []
    for nombre in MODELOS_DISPONIBLES:
        print(f"   {nombre:<15} ...", end="")
        try: res = _evaluar_individual(nombre, full_data, splits, final_seasonal_periods, metrica_opt)
        except Exception: res = None

        if res is not None:
            print(f" {metrica_opt}={res[metrica_opt.lower()]:.2f} | DAM={res['mae']:.2f} | Cobertura={res['picp']:.1f}%")
            results.append(res)
        else: print(" falló o sin datos")

    # CREACIÓN DEL SCORE GLOBAL NORMALIZADO (0 a 100)
    if results:
        # Usamos la métrica dinámica para calcular los máximos y mínimos del Score
        m_key = metrica_opt.lower()
        max_m, min_m = max(r[m_key] for r in results), min(r[m_key] for r in results)
        max_rmse, min_rmse = max(r['rmse'] for r in results), min(r['rmse'] for r in results)
        max_dam, min_dam = max(r['mae'] for r in results), min(r['mae'] for r in results)
        max_wink, min_wink = max(r['winkler'] for r in results), min(r['winkler'] for r in results)
        max_picp, min_picp = max(r['picp'] for r in results), min(r['picp'] for r in results)

        def norm(val, m_min, m_max, inverse=True):
            if m_max == m_min or not np.isfinite(val): return 0.0
            return (m_max - val)/(m_max - m_min) if inverse else (val - m_min)/(m_max - m_min)

        for r in results:
            n_m = norm(r[m_key], min_m, max_m, True)
            n_rmse = norm(r['rmse'], min_rmse, max_rmse, True)
            n_dam = norm(r['mae'], min_dam, max_dam, True)
            n_wink = norm(r['winkler'], min_wink, max_wink, True)
            n_picp = norm(r['picp'], min_picp, max_picp, False)

            # Promedio equitativo de los 5 factores (Normalizado a 100)
            r['score_global'] = ((n_m + n_rmse + n_dam + n_wink + n_picp) / 5.0) * 100.0

    # Ordenar por Score Global (El más cercano a 100 es el ganador absoluto)
    results.sort(key=lambda x: x['score_global'], reverse=True)
    return results

print("✅ Bloque 5 (Torneo con Métrica Dinámica S&OP) cargado correctamente.")

# =========================================================
# BLOQUE 6 Y 7 — PREPARACIÓN DEL MODELO FINAL, BOOTSTRAP Y PERCENTILES
# =========================================================
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

# --- FUNCIONES AUXILIARES FALTANTES RESTAURADAS ---
def _forecast_desde_resultado(model_info: dict, data: list, h: int) -> list:
    """Reconstruye el pronóstico del modelo ganador usando los parámetros óptimos del CV."""
    nombre = model_info.get('id', '')
    p = model_info.get('params', {})
    data_arr = np.asarray(data, dtype=float)

    try:
        if nombre == 'PROM_SIMPLE':
            return promedio_ventana_final(data_arr, h, len(data_arr))
        elif nombre == 'PROM_MOVIL':
            return promedio_ventana_final(data_arr, h, p.get('window', 1))
        elif nombre == 'PROM_PONDERADO':
            w = np.array(p.get('weights', [1.0]))
            win = p.get('window', len(w))
            if len(data_arr) < win: return [float(np.mean(data_arr))] * h
            w_norm = w / w.sum() if w.sum() > 0 else np.ones(win) / win
            pred = float(np.dot(data_arr[-win:], w_norm))
            return [pred] * h
        elif nombre == 'SEASONAL_NAIVE':
            return base_seasonal_naive(data_arr, h, p.get('seasonal_period', 2))
        elif nombre == 'SES':
            return base_ses(data_arr, h, alpha=p.get('alpha'))
        elif nombre == 'HOLT_ADD':
            return base_holt(data_arr, h, alpha=p.get('alpha'), beta=p.get('beta'), trend='add')
        elif nombre == 'HOLT_MUL':
            return base_holt(data_arr, h, alpha=p.get('alpha'), beta=p.get('beta'), trend='mul')
        elif nombre.startswith('HW_'):
            return base_holt_winters(data_arr, h,
                                     seasonal_periods=p.get('seasonal_periods', 2),
                                     alpha=p.get('alpha'), beta=p.get('beta'), gamma=p.get('gamma'),
                                     phi=p.get('phi'), trend=p.get('trend', 'add'),
                                     seasonal=p.get('seasonal', 'add'),
                                     damped_trend=p.get('damped_trend', False))
        elif nombre == 'ARIMA':
            return base_arima(data_arr, h, order=p.get('order', (0,0,0)),
                              seasonal_order=p.get('seasonal_order', (0,0,0,0)),
                              trend=p.get('trend', 'n'))
        elif nombre in ['RF', 'XGBOOST', 'LIGHTGBM']:
            mt = {'RF': 'rf', 'XGBOOST': 'xgb', 'LIGHTGBM': 'lgb'}[nombre]
            return train_ml_model(data_arr, h, model_type=mt, window=p.get('window', 3),
                                  model_params=p.get('model_params', {}),
                                  seasonal_periods=p.get('seasonal_periods', []))
    except Exception as e:
        print(f"⚠️ Error al reconstruir {nombre}: {e}")

    # Fallback seguro en caso de error extremo
    return promedio_ventana_final(data_arr, h, 1)

def _obtener_ajuste_train(model_info: dict, data: list) -> list:
    """
    Para S&OP, el riesgo se mide mejor con los errores Out-of-Sample (Rolling CV).
    Retornar NaN fuerza al algoritmo de Monte Carlo a usar el perfil de error 'cv_errors'
    calculado de manera rigurosa durante el torneo.
    """
    return [np.nan] * len(data)

# --- PREPARACIÓN DEL MODELO ---
def prepare_final_model(model_label: str, full_data: list, model_info: dict = None, uncertainty_horizon: int = 12):
    data = list(map(float, full_data))
    if model_info is None:
        model_info = {'id': 'PROM_SIMPLE', 'modelo': model_label, 'params': {'window': len(data)}}

    def forecast_func(d, h):
        return _forecast_desde_resultado(model_info, list(map(float, d)), int(h))

    error_profile = {
        'flat': model_info.get('cv_errors', [])
    }

    fitted = _obtener_ajuste_train(model_info, data)
    if not np.isnan(fitted).all():
        in_sample = (np.array(fitted) - np.array(data))
        error_profile['in_sample_errors'] = in_sample[np.isfinite(in_sample)].tolist()
    else:
        error_profile['in_sample_errors'] = []

    return forecast_func, error_profile, dict(model_info.get('params', {}))

# --- SIMULACIÓN MONTE CARLO ---
def monte_carlo_percentiles(
    forecast_func, full_data: list, k: int,
    residuos: dict, n_rep: int = 1000,
    perc_inf: float = 10.0, perc_sup: float = 90.0
):
    """Simulación Monte Carlo (Bootstrap Empírico) por Percentiles, libre de distribución."""
    base_forecast = np.asarray(forecast_func(full_data, k), dtype=float)
    nonnegative = np.all(np.asarray(full_data, dtype=float) >= 0)
    rng = np.random.default_rng(42)

    flat = np.asarray(residuos.get('flat', []), dtype=float)
    flat = flat[np.isfinite(flat)]
    in_sample = np.asarray(residuos.get('in_sample_errors', []), dtype=float)

    data_std = float(np.std(full_data, ddof=1)) if len(full_data) > 1 else 1.0
    max_allowed_std = data_std * 0.35

    if len(flat) >= 3:
        errores_base = flat
        method_used = 'Bootstrap Empírico (Out-of-Sample CV)'
    elif len(in_sample) >= 3:
        errores_base = in_sample
        method_used = 'Bootstrap Empírico (In-Sample)'
    else:
        errores_base = rng.normal(0, data_std * 0.15, 100)
        method_used = 'Normal Fallback (Falta de datos)'

    noise_errors = np.zeros((n_rep, k), dtype=float)

    for h in range(1, k + 1):
        growth_factor = np.sqrt(1 + 0.15 * (h - 1))
        muestras = rng.choice(errores_base, size=n_rep, replace=True)
        muestras_std = np.std(muestras)

        if muestras_std > max_allowed_std:
            muestras = muestras * (max_allowed_std / max(muestras_std, 1e-8))

        noise_errors[:, h - 1] = muestras * growth_factor

    sims = base_forecast.reshape(1, -1) - noise_errors
    if nonnegative:
        sims = np.maximum(sims, 0.0)

    mean_mc = sims.mean(axis=0)

    # OBTENCIÓN DE PERCENTILES
    p_lower = np.percentile(sims, perc_inf, axis=0)
    p_upper = np.percentile(sims, perc_sup, axis=0)

    index = [f"t+{i+1}" for i in range(k)]

    sim_stats = pd.DataFrame({
        "Base": base_forecast,
        "Media_MC": mean_mc,
        "Std_MC": sims.std(axis=0, ddof=1),
        "IC_Inferior": p_lower,
        "IC_Superior": p_upper
    }, index=index)
    replicas_df = pd.DataFrame(sims.T, index=index, columns=[f"Rep_{i+1}" for i in range(n_rep)])

    return base_forecast.tolist(), sim_stats, replicas_df, method_used, np.nan

print("✅ Bloques 6 y 7 (Funciones Auxiliares y Monte Carlo) cargados correctamente.")

"""
=========================================================
BLOQUE 8 — DASHBOARDS Y VISUALIZACIÓN
=========================================================
Genera interfaces visuales (Dashboards) para el diagnóstico de
la serie, la evaluación de la validación cruzada y el análisis
de los resultados de la simulación Monte Carlo.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf

def _save_fig(fig: plt.Figure, filename: str, title: str) -> None:
    """
    Guarda una figura de Matplotlib en formato PNG de alta calidad.

    Args:
        fig (plt.Figure): Objeto figura de Matplotlib.
        filename (str): Nombre base para el archivo (sin extensión).
        title (str): Título descriptivo para mostrar en consola.
    """
    out_dir = "resultados_dashboards"
    os.makedirs(out_dir, exist_ok=True)
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(filename))
    path = os.path.join(out_dir, f"{safe_name}.png")
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\n{title} GUARDADO EN:")
    print(f"   → {path}\n")

def dashboard_diagnostico_demanda(full_data: List[float],
                                  serie_name: str,
                                  window_size: int = 12,
                                  seasonal_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Genera el Dashboard 1: Diagnóstico integral de la demanda histórica.

    Analiza la tendencia lineal, distribución, coeficiente de variación,
    porcentaje de ceros y la función de autocorrelación (ACF) para
    clasificar el patrón dominante de la serie.

    Args:
        full_data (List[float]): Serie de tiempo histórica.
        serie_name (str): Nombre descriptivo de la serie (para títulos y archivo).
        window_size (int): Parámetro de compatibilidad (ya no se usa para forzar estacionalidad).
        seasonal_info (Dict, opcional): Información estacional extraída en el Bloque 3.

    Returns:
        str: Etiqueta descriptiva del patrón dominante encontrado
             (ej. 'ESTABLE', 'CON TENDENCIA', 'ESTACIONAL', 'ERRÁTICA', 'INTERMITENTE').
    """
    data = pd.Series(full_data, dtype=float)
    n = len(data)
    mean_val = data.mean()
    cv = (data.std() / abs(mean_val) * 100) if mean_val != 0 else 0
    zeros_pct = (data == 0).mean() * 100

    # Análisis de Tendencia
    X = np.arange(n).reshape(-1, 1)
    lr = LinearRegression().fit(X, data)
    slope = float(lr.coef_[0])
    rel_change = slope * max(1, n-1) / max(abs(mean_val), 1e-8)
    r2_trend = float(lr.score(X, data)) if n > 1 else 0.0
    min_r2 = float(AUTO_CONFIG.get('trend_min_r2', 0.10))

    if rel_change > 0.10:
        tendencia = "CRECIENTE" if r2_trend >= min_r2 else "CRECIENTE DÉBIL"
    elif rel_change < -0.10:
        tendencia = "DECRECIENTE" if r2_trend >= min_r2 else "DECRECIENTE DÉBIL"
    else:
        tendencia = "PLANA"

    # Análisis de Estacionalidad
    seasonal_info = seasonal_info or detectar_estacionalidad_automatica(full_data)
    candidatos = list(seasonal_info.get('candidatos', []))

    # Clasificación del Patrón Dominante
    if zeros_pct > 50:
        patron = "INTERMITENTE"
    elif candidatos:
        patron = "COMPLEJA (Estacional/Cíclica)" if tendencia != "PLANA" else "ESTACIONAL"
    elif cv > 50:
        patron = "ERRÁTICA"
    elif tendencia != "PLANA":
        patron = "CON TENDENCIA"
    else:
        patron = "ESTABLE"

    detrended = np.asarray(seasonal_info.get('detrended', data.values), dtype=float)

    # --- Renderizado Visual ---
    fig, axs = plt.subplots(3, 1, figsize=(16, 14))
    fig.suptitle(f"DIAGNÓSTICO DE DEMANDA – {serie_name}", fontsize=18, fontweight='bold')

    # Panel 1: Serie y Tendencia
    axs[0].plot(data.index, data.values, 'o-', color='teal')
    axs[0].plot(data.index, lr.predict(X), '--', color='red')
    axs[0].set_title("1. Serie histórica + Tendencia")
    axs[0].legend(["Demanda", "Tendencia"])
    axs[0].grid(True)

    # Panel 2: Distribución (Histograma nativo de matplotlib por estabilidad)
    axs[1].hist(data.values, bins=min(20, max(5, n//2)), alpha=0.75)
    axs[1].set_title(f"2. Distribución de la demanda (CV={cv:.1f}%, Zeros={zeros_pct:.1f}%)")

    # Panel 3: ACF (Autocorrelación) sin tendencia
    lags = min(40, max(1, n//2 - 1))
    plot_acf(detrended, lags=lags, ax=axs[2])
    for s in candidatos:
        if s <= lags:
            axs[2].axvline(s, color='red', linestyle='--', alpha=0.7)
    axs[2].set_title(f"3. ACF sin tendencia | N candidatos: {candidatos if candidatos else 'ninguno'}")

    plt.tight_layout()
    _save_fig(fig, f"dashboard1_diagnostico_{serie_name}", "DASHBOARD 1 – DIAGNÓSTICO")

    print("\nRESUMEN DIAGNÓSTICO")
    print(f"   Patrón dominante : {patron}")
    print(f"   Tendencia        : {tendencia}")
    print(f"   Fuerza tendencia : R²={r2_trend:.3f}")
    print(f"   N candidatos     : {candidatos if candidatos else 'ninguno'}")
    print(f"   CV               : {cv:.1f}%")
    print(f"   % de ceros       : {zeros_pct:.1f}%")
    print(f"   Períodos (n)     : {n}")

    return patron

def dashboard_evaluacion_modelos(eval_results: List[Dict[str, Any]],
                                 test_data: List[float],
                                 train_data: List[float],
                                 serie_name: str,
                                 u: int) -> None:
    """
    Genera el Dashboard 2: Resultados del Rolling-Origin Cross-Validation.

    Muestra cómo se comportaron los 3 mejores modelos prediciendo
    los datos reales del último pliegue (fold) de validación, y
    la distribución general de sus errores fuera de muestra.

    Args:
        eval_results (List[Dict]): Resultados de evaluación de todos los modelos (ordenados).
        test_data (List[float]): Datos reales del último pliegue de prueba.
        train_data (List[float]): Datos reales del último pliegue de entrenamiento.
        serie_name (str): Nombre descriptivo de la serie.
        u (int): Horizonte de validación.
    """
    if not eval_results:
        print("⚠️ No hay resultados para mostrar en el dashboard de evaluación.")
        return

    test = np.asarray(test_data, dtype=float)
    train = np.asarray(train_data, dtype=float)
    n_train, n_test = len(train), len(test)
    metric_name = eval_results[0].get('metrica_principal', 'MAPE')

    df_sorted = pd.DataFrame(eval_results).sort_values('score_cv')
    top3 = df_sorted.iloc[:3]

    fig, axes = plt.subplots(3, 1, figsize=(18, 18))
    fig.suptitle(f"Evaluación de Modelos – Rolling CV\nSerie: {serie_name} | horizonte fold={u}",
                 fontsize=20, fontweight="bold")

    # Panel 1: Ajuste del Pronóstico (Último Fold)
    axes[0].plot(range(n_train), train, ':', color='gray', label='Train último fold')
    axes[0].plot(range(n_train, n_train+n_test), test, 'o-', color='black', label='Test último fold')
    axes[0].axvline(n_train-0.5, color='red', linestyle='--')
    for _, row in top3.iterrows():
        preds = np.asarray(row['predictions'], dtype=float)
        axes[0].plot(range(n_train, n_train+len(preds)), preds, 'x--',
                     label=f"{row['modelo']} ({metric_name} CV={row['score_cv']:.2f})")
    axes[0].set_title("1. Pronósticos vs datos no vistos del último fold (TOP 3)")
    axes[0].set_xlabel("Periodo")
    axes[0].set_ylabel("Demanda")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    # Panel 2: Curvas de Error (Convención e = F - D)
    axes[1].axhline(0, color='black', linewidth=1.5)
    for _, row in top3.iterrows():
        preds = np.asarray(row['predictions'], dtype=float)
        errors = preds - test[:len(preds)]
        axes[1].plot(range(1, len(errors)+1), errors, marker='o', linewidth=2,
                     label=f"{row['modelo']} (Bias={np.mean(errors):.2f})")
    axes[1].set_title("2. Curvas de error e = F - D en el último fold")
    axes[1].set_xlabel("Horizonte dentro del fold")
    axes[1].set_ylabel("Error")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    # Panel 3: Distribución global de errores out-of-sample (Boxplot)
    errores_dict = {row['modelo']: pd.Series(row.get('cv_errors', [])) for _, row in df_sorted.iterrows()}
    df_err = pd.DataFrame(errores_dict)
    if not df_err.empty:
        clean_cols = [c for c in df_err.columns if df_err[c].dropna().shape[0] > 0]
        if clean_cols:
            axes[2].boxplot([df_err[c].dropna().values for c in clean_cols], labels=clean_cols)
            axes[2].tick_params(axis='x', rotation=35)

    axes[2].axhline(0, color='black', linewidth=1.5)
    axes[2].set_title("3. Distribución de errores fuera de muestra por modelo")
    axes[2].set_xlabel("Modelo")
    axes[2].set_ylabel("Error = Predicción - Real")
    axes[2].grid(True)

    plt.tight_layout()
    _save_fig(fig, f"dashboard2_evaluacion_{serie_name}", "DASHBOARD 2 – COMPARACIÓN + ERRORES")

def dashboard_pronostico_mc(full_data: List[float],
                            sim_stats: pd.DataFrame,
                            replicas: pd.DataFrame,
                            serie_name: str,
                            modelo: str,
                            k: int,
                            u: int,
                            method_used: str = "auto",
                            z_value: float = np.nan,
                            residuos: Optional[Dict[str, Any]] = None) -> None:
    """
    Genera el Dashboard 3: Pronóstico Oficial y Análisis Monte Carlo.
    Diseñado con estética corporativa y barras de error (Mínimos y Máximos).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display

    n = len(full_data)
    idx_hist = range(n)
    idx_futuro = range(n, n + k)

    # 1. Extracción de límites
    base = sim_stats['Base'].values
    lower = sim_stats['IC_Inferior'].values
    upper = sim_stats['IC_Superior'].values

    # 2. Configuración del Lienzo (1 fila, 2 columnas)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    fig.suptitle(f"PROYECCIÓN Y ANÁLISIS DE RIESGO DE INVENTARIO\nSerie: {serie_name} | Modelo: {modelo}",
                 fontsize=15, fontweight='bold', color='#333333')

    # ==========================================
    # PANEL 1: Pronóstico con Mínimos y Máximos
    # ==========================================
    # Historia
    ax1.plot(idx_hist, full_data, color='gray', marker='o', linestyle='-', linewidth=2, label="Demanda Histórica")

    # Sombreado suave de fondo para el área de riesgo
    ax1.fill_between(idx_futuro, lower, upper, color='#D32F2F', alpha=0.10, label='Zona de Riesgo')

    # "El puntico con las líneas de mínimos y máximos" (Error Bars)
    yerr_lower = base - lower
    yerr_upper = upper - base
    ax1.errorbar(idx_futuro, base, yerr=[yerr_lower, yerr_upper],
                 fmt='s', color='#D32F2F', ecolor='#D32F2F', elinewidth=2, capsize=6, capthick=2,
                 markersize=6, linestyle='--', linewidth=1.5, label="Pronóstico Base + Topes (Min/Max)")

    ax1.axvline(n - 0.5, color='orange', linestyle='--', linewidth=1.5)
    ax1.set_title("1. Pronóstico Oficial e Intervalos de Confianza", fontsize=13, fontweight='bold')
    ax1.set_xlabel("Período (t)")
    ax1.set_ylabel("Unidades Demandadas")
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper left')

    # ==========================================
    # PANEL 2: Rayos X de la Simulación (Spaghetti)
    # ==========================================
    max_traj = min(100, replicas.shape[1]) # Mostrar hasta 100 escenarios grises
    for i in range(max_traj):
        ax2.plot(idx_futuro, replicas.iloc[:, i], color='gray', alpha=0.10)

    ax2.plot(idx_futuro, base, color='#D32F2F', marker='s', linestyle='-', linewidth=2, label='Pronóstico Base')
    ax2.plot(idx_futuro, lower, color='black', linestyle=':', linewidth=2, label='Peor Escenario (Min)')
    ax2.plot(idx_futuro, upper, color='black', linestyle=':', linewidth=2, label='Mejor Escenario (Max)')

    ax2.set_title(f"2. Escenarios Simulados ({max_traj} de {replicas.shape[1]})", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Período Futuro")
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend(loc='upper left')

    plt.tight_layout()

    # 3. Mostrar y Guardar
    _save_fig(fig, f"dashboard3_pronostico_{serie_name}", "DASHBOARD 3 – PRONÓSTICO + MC")
    plt.show()  # Esto asegura que la gráfica renderice directamente en el notebook de Colab

# =========================================================
# BLOQUE 9 — EXPORTACIÓN A CSV Y GUARDADO DE EXPERIMENTO
# =========================================================
import os
import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

def _json_safe(obj: Any) -> Any:
    """Convierte objetos complejos (tuplas, arreglos de NumPy) a formatos seguros para JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj) if np.isfinite(obj) else None
    return obj

def guardar_experimento_csv(serie_name: str,
                            modelo_final: str,
                            u: int,
                            k: int,
                            n_rep: int,
                            base_forecast: List[float],
                            df_stats: pd.DataFrame,
                            df_replicas: pd.DataFrame,
                            resultados_evaluacion: List[Dict[str, Any]],
                            escala: int = 1,
                            final_params: Optional[Dict[str, Any]] = None,
                            seasonal_info: Optional[Dict[str, Any]] = None) -> str:
    """
    Guarda los resultados del experimento en la carpeta 'resultados_dashboards'.
    """
    out_dir = "resultados_dashboards"
    os.makedirs(out_dir, exist_ok=True)
    index = [f"t+{i+1}" for i in range(k)]
    df_stats = df_stats.reindex(index)
    params_json = json.dumps(_json_safe(final_params or {}), ensure_ascii=False)

    df_exp = pd.DataFrame({
        'Serie': [serie_name]*k,
        'Modelo': [modelo_final]*k,
        'Escala': [escala]*k,
        'u_CV': [u]*k,
        'k': [k]*k,
        'Periodo': index,
        'Base': base_forecast,
        'Media_MC': df_stats['Media_MC'].values,
        'Std_MC': df_stats['Std_MC'].values,
        'IC_Inferior': df_stats['IC_Inferior'].values,
        'IC_Superior': df_stats['IC_Superior'].values,
        'Parametros_Finales': [params_json]*k,
    })

    for c in ['Bias_CV_H', 'Std_Error_H', 'N_Errores_H']:
        if c in df_stats:
            df_exp[c] = df_stats[c].values

    df_exp = pd.concat([df_exp, df_replicas.reset_index(drop=True)], axis=1)

    clean_series = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(serie_name))
    base_name = f"resultados_experimento_{clean_series}_{datetime.now().strftime('%Y%m%d')}"
    csv_path = os.path.join(out_dir, base_name + ".csv")
    df_exp.to_csv(csv_path, index=False)

    # Exportar resumen de la evaluación de todos los modelos del torneo
    if resultados_evaluacion:
        rows = []
        for r in resultados_evaluacion:
            rows.append({
                'modelo': r.get('modelo'), 'id': r.get('id'), 'metrica_principal': r.get('metrica_principal'),
                'score_cv': r.get('score_cv'), 'mape': r.get('mape'), 'mae': r.get('mae'),
                'mse': r.get('mse'), 'rmse': r.get('rmse'), 'mase': r.get('mase'), 'bias': r.get('bias'),
                'n_folds': r.get('n_folds'),
                'params': json.dumps(_json_safe(r.get('params', {})), ensure_ascii=False)
            })
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, base_name + "_evaluacion.csv"), index=False)

    print(f"\n📁 EXPERIMENTO EXPORTADO A CSV:\n   → {csv_path}\n")
    return csv_path

print("✅ Bloque 9 (Exportación a CSV) cargado correctamente.")


# =========================================================================
# 🚀 INTERFAZ DE USUARIO WEB (Reemplaza los Bloques 10 y 11 de Colab)
# =========================================================================

st.title("📊 Herramienta de Pronósticos S&OP")
st.markdown("Sube tu archivo histórico, ajusta los parámetros en el menú lateral y genera los pronósticos optimizados sin tocar una sola línea de código.")

# 1. MENÚ LATERAL: CARGA DE DATOS
st.sidebar.header("1. Carga de Datos")
archivo_subido = st.sidebar.file_uploader("Sube tu archivo (Excel o CSV)", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    # Cargar datos según extensión
    if archivo_subido.name.endswith('.csv'):
        df = pd.read_csv(archivo_subido)
    else:
        df = pd.read_excel(archivo_subido)
    
    # Selector de columna
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    col_demanda = st.sidebar.selectbox("➤ Selecciona la columna de Demanda:", columnas_numericas)
    
    # Extraer la serie limpia
    data = df[col_demanda].dropna().values
    n_total = len(data)
    
    st.write("### Comportamiento Histórico de la Demanda")
    fig_inicial, ax_inicial = plt.subplots(figsize=(12, 4))
    ax_inicial.plot(range(1, n_total + 1), data, marker='o', color='#1F4E79')
    ax_inicial.set_xlabel("Período")
    ax_inicial.set_ylabel("Demanda")
    ax_inicial.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig_inicial)

    # 2. MENÚ LATERAL: CONFIGURACIÓN
    st.sidebar.header("2. Configuración del Torneo")
    train_size = st.sidebar.number_input("Datos de Entrenamiento (Train)", min_value=5, max_value=n_total-1, value=int(n_total*0.8))
    h_futuro = st.sidebar.number_input("Horizonte a Proyectar (k meses)", min_value=1, value=12)
    estacionalidad = st.sidebar.number_input("Estacionalidad a forzar (N)", min_value=2, value=12)

    # 3. BOTÓN DE EJECUCIÓN
    if st.sidebar.button("🚀 Ejecutar Torneo de Modelos"):
        
        # Muestra un spinner o barra de carga mientras corre los modelos
        with st.spinner('Entrenando algoritmos, optimizando parámetros y evaluando riesgo... Esto puede tomar un minuto.'):
            
            # --- Aquí llamas a tu función evaluar_modelos_seleccionados ---
            # results = evaluar_modelos_seleccionados(data.tolist(), seasonal_periods_hw=[int(estacionalidad)], cv_horizon=(n_total-train_size), min_train=train_size)
            
            st.success("¡Torneo completado con éxito!")
            
            st.markdown("### 🏆 Ranking de Algoritmos (Score S&OP)")
            # st.dataframe(pd.DataFrame(results)) # Muestra la tabla de ranking en pantalla
            
            # st.markdown(f"**Campeón Absoluto:** {results[0]['modelo']}")
            
            # Aquí puedes llamar a tu función de gráficos pasándole st.pyplot(fig) en lugar de plt.show()
            # Y podrías imprimir los parámetros optimizados usando st.write() en lugar de print()
            
else:
    st.info("👈 Por favor, sube un archivo en el panel izquierdo para comenzar.")
