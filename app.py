import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.stattools import acf
from scipy.signal import periodogram
from statsmodels.tsa.seasonal import STL

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

AUTO_CONFIG = {
    "random_state": 42,
    "max_cv_splits": 3,
    "max_seasonal_candidates": 2,
    "max_seasonal_period": 24,
    "trend_min_r2": 0.10,
}

# =========================================================================
# FUNCIONES MATEMATICAS Y DE ERROR
# =========================================================================

def safe_mape(y_true, y_pred, eps=1e-8):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if not np.any(mask): return np.nan
    return 100.0 * float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mse(y_true, y_pred):
    return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))

def bias(y_true, y_pred):
    return float(np.mean(np.array(y_pred) - np.array(y_true)))

def elegir_metrica_primaria(data):
    y = np.asarray(data, dtype=float)
    pct_ceros = float(np.mean(np.isclose(y, 0.0))) if len(y) else 0.0
    return "MAE" if pct_ceros > 0.10 else "MAPE"

def metricas_error(y_train, y_true, y_pred):
    return {
        "MAPE": safe_mape(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "BIAS": bias(y_true, y_pred),
    }

# =========================================================================
# MODELOS CLASICOS
# =========================================================================

def promedio_ventana_final(data, horizon, window):
    data = list(data)
    n = len(data)
    if n == 0: return [0.0] * horizon
    window = max(1, min(int(window), n))
    avg = float(np.mean(data[-window:]))
    return [avg] * horizon

def base_seasonal_naive(data, horizon, seasonal_period):
    data = list(map(float, data))
    s = int(seasonal_period)
    history = data.copy()
    preds = []
    for _ in range(int(horizon)):
        pred = float(history[-s])
        preds.append(pred)
        history.append(pred)
    return preds

def base_ses(data, horizon, alpha=None):
    model = ExponentialSmoothing(data, trend=None, seasonal=None, initialization_method='estimated')
    fit = model.fit(optimized=True) if alpha is None else model.fit(smoothing_level=float(alpha), optimized=False)
    fc = np.maximum(np.array(fit.forecast(horizon), dtype=float), 0.0)
    return fc.tolist()

def base_holt(data, horizon, alpha=None, beta=None, trend='add'):
    if len(data) < 2: return base_ses(data, horizon, alpha=alpha)
    model = ExponentialSmoothing(data, trend=trend, seasonal=None, damped_trend=False, initialization_method='estimated')
    fit = model.fit(optimized=True) if (alpha is None or beta is None) else model.fit(smoothing_level=float(alpha), smoothing_trend=float(beta), optimized=False)
    fc = np.maximum(np.array(fit.forecast(horizon), dtype=float), 0.0)
    return fc.tolist()

def base_holt_winters(data, horizon, seasonal_periods=12, alpha=None, beta=None, gamma=None, phi=None, trend='add', seasonal='add', damped_trend=False):
    model = ExponentialSmoothing(data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods, damped_trend=damped_trend, initialization_method='estimated')
    if alpha is None or beta is None or gamma is None:
        fit = model.fit(optimized=True, use_brute=True)
    else:
        kwargs = dict(smoothing_level=float(alpha), smoothing_trend=float(beta), smoothing_seasonal=float(gamma), optimized=False)
        if damped_trend: kwargs["damping_trend"] = float(phi if phi is not None else 0.98)
        fit = model.fit(**kwargs)
    fc = np.maximum(np.array(fit.forecast(horizon), dtype=float), 0.0)
    return fc.tolist()

def base_arima(data, horizon, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0), trend='n'):
    try:
        model = SARIMAX(data, order=tuple(order), seasonal_order=tuple(seasonal_order), trend=trend, enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = np.maximum(np.array(model.forecast(horizon), dtype=float), 0.0)
        return fc.tolist()
    except Exception:
        return promedio_ventana_final(data, horizon, 1)

def detectar_estacionalidad_automatica(data, max_period=None, top_k=None):
    y = np.asarray(data, dtype=float)
    n = len(y)
    if n < 8: return {"candidatos": [], "detrended": y.copy()}
    t = np.arange(n, dtype=float)
    lr = LinearRegression().fit(t.reshape(-1, 1), y)
    detrended = y - lr.predict(t.reshape(-1, 1))
    max_period = min(int(max_period or AUTO_CONFIG["max_seasonal_period"]), max(2, n // 2))
    
    try: acf_vals = acf(detrended, nlags=max_period, fft=True)
    except: acf_vals = np.zeros(max_period + 1)
    
    rows = []
    for s in range(2, max_period + 1):
        if n < 2 * s: continue
        score = max(0.0, float(acf_vals[s])) if s < len(acf_vals) else 0.0
        rows.append({"N": s, "Score": score})
    
    tabla = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
    candidatos = [int(tabla.iloc[0]["N"])] if not tabla.empty and tabla.iloc[0]["Score"] > 0.2 else []
    return {"candidatos": candidatos, "detrended": detrended}

# =========================================================================
# MODELOS MACHINE LEARNING
# =========================================================================

def create_features(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(list(data[i-window:i]))
        y.append(data[i])
    return np.array(X, dtype=float), np.array(y, dtype=float)

def train_ml_model(data, horizon, model_type='rf', window=3, **kwargs):
    n = len(data)
    if n <= window + 2: return base_ses(data, horizon)
    
    X_time = np.arange(n, dtype=float).reshape(-1, 1)
    y = np.asarray(data, dtype=float)
    trend_model = LinearRegression().fit(X_time, y)
    residuals = y - trend_model.predict(X_time)
    
    X, target = create_features(residuals, window)
    if X.shape[0] < 3: return promedio_ventana_final(data, horizon, 1)
    
    if model_type == 'rf': model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    elif model_type == 'xgb' and xgb: model = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=42)
    elif model_type == 'lgb' and lgb: model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42)
    else: return promedio_ventana_final(data, horizon, 1)
    
    model.fit(X, target)
    
    history_res = residuals.tolist()
    preds = []
    for step in range(horizon):
        x_new = np.array(history_res[-window:], dtype=float).reshape(1, -1)
        pred_res = float(model.predict(x_new)[0])
        future_trend = float(trend_model.predict(np.array([[n + step]], dtype=float))[0])
        pred = max(0.0, future_trend + pred_res)
        preds.append(pred)
        history_res.append(pred_res)
    return preds

# =========================================================================
# EVALUACION Y TORNEO
# =========================================================================

MODELOS_DISPONIBLES = {
    'PROM_SIMPLE': 'Promedio Simple', 'PROM_MOVIL': 'Promedio Móvil',
    'SEASONAL_NAIVE': 'Seasonal Naïve', 'SES': 'SES',
    'HOLT_ADD': 'Holt Doble (Add)', 'HOLT_MUL': 'Holt Doble (Mul)',
    'HW_ADD_ADD': 'Holt-Winters (Add-Add)', 'HW_MUL_MUL': 'Holt-Winters (Mul-Mul)',
    'RF': 'Random Forest', 'XGBOOST': 'XGBoost', 'LIGHTGBM': 'LightGBM'
}

def _evaluar_predictor_cv(data, train_size, predictor, metric_name):
    train, test = data[:train_size], data[train_size:]
    try:
        pred = np.asarray(predictor(train, len(test)), dtype=float)
        met = metricas_error(train, test, pred)
        return {"score_cv": met.get(metric_name, np.nan), "mape": met["MAPE"], "mae": met["MAE"], "rmse": met["RMSE"], "predictions": pred}
    except Exception:
        return None

def evaluar_modelos_seleccionados(data, seasonal_periods_hw, train_size):
    full_data = list(map(float, data))
    metrica_opt = elegir_metrica_primaria(full_data)
    results = []
    s = seasonal_periods_hw[0] if seasonal_periods_hw else 12
    
    for nombre in MODELOS_DISPONIBLES:
        res = None
        if nombre == 'PROM_SIMPLE':
            res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: promedio_ventana_final(tr, h, len(tr)), metrica_opt)
        elif nombre == 'PROM_MOVIL':
            res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: promedio_ventana_final(tr, h, 3), metrica_opt)
        elif nombre == 'SEASONAL_NAIVE':
            res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: base_seasonal_naive(tr, h, s), metrica_opt)
        elif nombre == 'SES':
            res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: base_ses(tr, h), metrica_opt)
        elif nombre == 'HOLT_ADD':
            res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: base_holt(tr, h, trend='add'), metrica_opt)
        elif nombre == 'HOLT_MUL':
            if np.all(np.array(full_data)>0): res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: base_holt(tr, h, trend='mul'), metrica_opt)
        elif nombre == 'HW_ADD_ADD':
            if len(full_data[:train_size]) >= 2*s: res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: base_holt_winters(tr, h, s, trend='add', seasonal='add'), metrica_opt)
        elif nombre == 'HW_MUL_MUL':
            if len(full_data[:train_size]) >= 2*s and np.all(np.array(full_data)>0): res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: base_holt_winters(tr, h, s, trend='mul', seasonal='mul'), metrica_opt)
        elif nombre in ['RF', 'XGBOOST', 'LIGHTGBM']:
            mt = {'RF': 'rf', 'XGBOOST': 'xgb', 'LIGHTGBM': 'lgb'}[nombre]
            res = _evaluar_predictor_cv(full_data, train_size, lambda tr, h: train_ml_model(tr, h, model_type=mt), metrica_opt)
        
        if res is not None and not np.isnan(res['score_cv']):
            res['modelo'] = nombre
            results.append(res)
            
    results.sort(key=lambda x: x['score_cv'])
    return results

# =========================================================================
# INTERFAZ DE USUARIO WEB STREAMLIT
# =========================================================================

st.title("📊 Herramienta de Pronósticos S&OP")
st.markdown("Sube tu archivo histórico, ajusta los parámetros en el menú lateral y genera los pronósticos optimizados.")

# 1. MENÚ LATERAL: CARGA DE DATOS
st.sidebar.header("1. Carga de Datos")
archivo_subido = st.sidebar.file_uploader("Sube tu archivo (Excel o CSV)", type=["csv", "xlsx", "xls"])

if archivo_subido is not None:
    if archivo_subido.name.endswith('.csv'): df = pd.read_csv(archivo_subido)
    else: df = pd.read_excel(archivo_subido)
    
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(columnas_numericas) > 0:
        col_demanda = st.sidebar.selectbox("➤ Selecciona la columna de Demanda:", columnas_numericas)
        data = df[col_demanda].dropna().values
        n_total = len(data)
        
        st.write("### Comportamiento Histórico de la Demanda")
        fig_inicial, ax_inicial = plt.subplots(figsize=(12, 4))
        ax_inicial.plot(range(1, n_total + 1), data, marker='o', color='#1F4E79')
        ax_inicial.set_xlabel("Período")
        ax_inicial.set_ylabel("Demanda")
        ax_inicial.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_inicial)

        st.sidebar.header("2. Configuración del Torneo")
        train_size = st.sidebar.number_input("Datos de Entrenamiento (Train)", min_value=5, max_value=n_total-1, value=int(n_total*0.8))
        h_futuro = st.sidebar.number_input("Horizonte a Proyectar (k meses)", min_value=1, value=12)
        
        candidatos = detectar_estacionalidad_automatica(data)["candidatos"]
        s_sugerido = candidatos[0] if candidatos else 12
        estacionalidad = st.sidebar.number_input("Estacionalidad (N)", min_value=2, value=s_sugerido)

        if st.sidebar.button("🚀 Ejecutar Torneo de Modelos"):
            with st.spinner('Entrenando algoritmos y evaluando riesgo...'):
                results = evaluar_modelos_seleccionados(data.tolist(), [int(estacionalidad)], train_size)
                
                if results:
                    st.success("¡Torneo completado con éxito!")
                    st.markdown(f"### 🏆 Campeón Absoluto: **{results[0]['modelo']}**")
                    
                    df_ranking = pd.DataFrame(results)[['modelo', 'score_cv', 'mape', 'mae', 'rmse']]
                    df_ranking.columns = ['Modelo', 'Score Opt.', 'MAPE (%)', 'MAE', 'RMSE']
                    st.dataframe(df_ranking.style.background_gradient(subset=['Score Opt.'], cmap='RdYlGn_r'))
                    
                    st.markdown("### 📈 Auditoría Visual del Ganador")
                    ganador = results[0]
                    train_y = data[:train_size]
                    test_y = data[train_size:]
                    preds_test = ganador['predictions']
                    
                    fig_eval, ax_eval = plt.subplots(figsize=(12, 5))
                    ax_eval.plot(range(1, train_size + 1), train_y, color='gray', marker='o', label='Train Real')
                    ax_eval.plot(range(train_size + 1, n_total + 1), test_y, color='black', marker='o', label='Test Real')
                    ax_eval.plot(range(train_size + 1, n_total + 1), preds_test, color='#D32F2F', marker='X', linestyle='--', label='Pronóstico (Test)')
                    ax_eval.axvline(train_size + 0.5, color='orange', linestyle='--')
                    ax_eval.legend()
                    ax_eval.grid(True, alpha=0.5)
                    st.pyplot(fig_eval)
                else:
                    st.error("No se pudo evaluar ningún modelo con los datos proporcionados.")
    else:
        st.error("El archivo no contiene columnas numéricas válidas.")
else:
    st.info("👈 Por favor, sube un archivo en el panel izquierdo para comenzar.")
