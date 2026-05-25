"""
generate_analysis.py
Computa el análisis completo de series temporales y genera index.html
Diego Garcia Alvear — Métodos II, Ordinaria 2026
"""
import json, math, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

# ── 1. DATOS ──────────────────────────────────────────────────────────────────
# El CSV usa coma como separador decimal (formato europeo, read.csv2 en R).
# Lectura correcta: reemplazar coma decimal por punto.
with open("Ordinaria_DGAO_datosR.csv", "r", encoding="utf-8") as _f:
    _lines = _f.readlines()
valores = np.array([
    float(line.strip().replace(",", "."))
    for line in _lines[1:]   # saltar cabecera
    if line.strip()
])
n = len(valores)
t_index = np.arange(n) / 7.0  # índice temporal como en R: start=c(1,3), step=1/7

# ── 2. ESTADÍSTICOS BÁSICOS ───────────────────────────────────────────────────
kpi = {
    "n": int(n),
    "mean": float(np.mean(valores)),
    "std": float(np.std(valores, ddof=1)),
    "min": float(np.min(valores)),
    "max": float(np.max(valores)),
    "median": float(np.median(valores)),
}

# ── 3. DESCOMPOSICIÓN ─────────────────────────────────────────────────────────
decomp = seasonal_decompose(valores, model="additive", period=7, extrapolate_trend="freq")
trend_vals     = decomp.trend.tolist()
seasonal_vals  = decomp.seasonal.tolist()
residual_vals  = decomp.resid.tolist()

# ── 4. TRANSFORMACIÓN LOG ─────────────────────────────────────────────────────
log_vals = np.sign(valores) * np.log10(np.abs(valores))
dlog_vals = np.diff(log_vals)   # primera diferencia (len = n-1)

# ── 5. ADF TESTS ──────────────────────────────────────────────────────────────
adf_orig  = adfuller(valores, autolag="AIC")
adf_dlog  = adfuller(dlog_vals, autolag="AIC")

# ── 6. ACF / PACF (lag_max = 100) ────────────────────────────────────────────
nlags = 100
ci_band = 1.96 / math.sqrt(n)

acf_orig_vals  = acf(valores,   nlags=nlags, fft=True).tolist()
pacf_orig_vals = pacf(valores,  nlags=nlags, method="ywm").tolist()
acf_dlog_vals  = acf(dlog_vals, nlags=nlags, fft=True).tolist()
pacf_dlog_vals = pacf(dlog_vals,nlags=nlags, method="ywm").tolist()

# ── 7. MODELOS ARIMA ──────────────────────────────────────────────────────────
print("Ajustando Modelo 1 SARIMA(1,1,0)(0,1,1,7)...")
mod1 = SARIMAX(valores, order=(1,1,0), seasonal_order=(0,1,1,7),
               enforce_stationarity=False, enforce_invertibility=False)
res1 = mod1.fit(disp=False, maxiter=200)

print("Ajustando Modelo 2 SARIMA(2,1,0)(2,1,0,7) sobre dlog...")
mod2 = SARIMAX(dlog_vals, order=(2,1,0), seasonal_order=(2,1,0,7),
               enforce_stationarity=False, enforce_invertibility=False)
res2 = mod2.fit(disp=False, maxiter=200)

fitted1 = res1.fittedvalues.tolist()
fitted2 = res2.fittedvalues.tolist()
resid1  = res1.resid.tolist()
resid2  = res2.resid.tolist()

# Coeficientes
def model_summary(res, name):
    param_names = res.param_names if hasattr(res, 'param_names') else [f"p{i}" for i in range(len(res.params))]
    params = {str(k): float(v) for k, v in zip(param_names, res.params)}
    pvals  = {str(k): float(v) for k, v in zip(param_names, res.pvalues)}
    return {
        "name": name,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "llf": float(res.llf),
        "params": params,
        "pvalues": pvals,
    }

m1_sum = model_summary(res1, "SARIMA(1,1,0)(0,1,1)₇ — Serie original")
m2_sum = model_summary(res2, "SARIMA(2,1,0)(2,1,0)₇ — Serie dlog")

# ── 8. LJUNG-BOX ─────────────────────────────────────────────────────────────
lb1 = acorr_ljungbox(resid1, lags=[10], return_df=True)
lb2 = acorr_ljungbox(resid2, lags=[10], return_df=True)
lb_pval1 = float(lb1["lb_pvalue"].iloc[0])
lb_pval2 = float(lb2["lb_pvalue"].iloc[0])

# ── 9. FORECAST 4 PERIODOS ───────────────────────────────────────────────────
fc1 = res1.get_forecast(steps=4)
fc1_mean = fc1.predicted_mean.tolist()
_ci1_80 = fc1.conf_int(alpha=0.20)
_ci1_90 = fc1.conf_int(alpha=0.10)
fc1_ci80 = (_ci1_80.values if hasattr(_ci1_80, 'values') else _ci1_80).tolist()
fc1_ci90 = (_ci1_90.values if hasattr(_ci1_90, 'values') else _ci1_90).tolist()

fc2 = res2.get_forecast(steps=4)
fc2_mean = fc2.predicted_mean.tolist()
_ci2_80 = fc2.conf_int(alpha=0.20)
_ci2_90 = fc2.conf_int(alpha=0.10)
fc2_ci80 = (_ci2_80.values if hasattr(_ci2_80, 'values') else _ci2_80).tolist()
fc2_ci90 = (_ci2_90.values if hasattr(_ci2_90, 'values') else _ci2_90).tolist()

# ── 10. SERIALIZAR TODO ───────────────────────────────────────────────────────
data = {
    "kpi": kpi,
    "t_index": t_index.tolist(),
    "valores": valores.tolist(),
    "decomp": {
        "trend":    trend_vals,
        "seasonal": seasonal_vals,
        "residual": residual_vals,
    },
    "log_vals":  log_vals.tolist(),
    "dlog_vals": dlog_vals.tolist(),
    "adf": {
        "orig":  {"stat": float(adf_orig[0]), "pval": float(adf_orig[1])},
        "dlog":  {"stat": float(adf_dlog[0]), "pval": float(adf_dlog[1])},
    },
    "acf_pacf": {
        "nlags": nlags,
        "ci_band": ci_band,
        "acf_orig":  acf_orig_vals,
        "pacf_orig": pacf_orig_vals,
        "acf_dlog":  acf_dlog_vals,
        "pacf_dlog": pacf_dlog_vals,
    },
    "models": {
        "m1": m1_sum,
        "m2": m2_sum,
        "fitted1": fitted1,
        "fitted2": fitted2,
        "resid1":  resid1,
        "resid2":  resid2,
        "lb_pval1": lb_pval1,
        "lb_pval2": lb_pval2,
    },
    "forecast": {
        "m1": {"mean": fc1_mean, "ci80": fc1_ci80, "ci90": fc1_ci90},
        "m2": {"mean": fc2_mean, "ci80": fc2_ci80, "ci90": fc2_ci90},
    },
}

print("Análisis completado. Generando HTML...")


def build_html(data_json: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Series Temporales — Diego Garcia Alvear</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --bg: #0f1117; --bg2: #181c27; --bg3: #1e2336;
    --border: #2a3050; --accent: #4f8ef7; --accent2: #f7c948;
    --red: #f76b6b; --blue: #4f8ef7; --green: #4fcc8e;
    --text: #e4e8f0; --muted: #7a85a0;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Inter', sans-serif; font-size: 14px; }}
  header {{
    background: linear-gradient(135deg, #1a2040 0%, #0f1530 100%);
    border-bottom: 1px solid var(--border);
    padding: 20px 32px;
    display: flex; align-items: center; justify-content: space-between;
  }}
  header h1 {{ font-size: 20px; font-weight: 700; color: #fff; }}
  header h1 span {{ color: var(--accent2); }}
  header p {{ color: var(--muted); font-size: 12px; margin-top: 4px; }}
  .badge {{
    background: var(--accent); color: #fff; font-size: 11px; font-weight: 600;
    padding: 4px 10px; border-radius: 20px;
  }}
  .kpi-row {{
    display: flex; gap: 16px; padding: 20px 32px; flex-wrap: wrap;
    background: var(--bg2); border-bottom: 1px solid var(--border);
  }}
  .kpi-card {{
    flex: 1; min-width: 120px; background: var(--bg3);
    border: 1px solid var(--border); border-radius: 10px;
    padding: 14px 18px;
  }}
  .kpi-card .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }}
  .kpi-card .value {{ font-size: 22px; font-weight: 700; color: var(--accent); margin-top: 4px; }}
  .tabs-bar {{
    display: flex; gap: 4px; padding: 12px 32px 0;
    background: var(--bg2); border-bottom: 1px solid var(--border);
    position: sticky; top: 0; z-index: 10; overflow-x: auto;
  }}
  .tab-btn {{
    padding: 8px 16px; border-radius: 8px 8px 0 0; border: 1px solid transparent;
    background: transparent; color: var(--muted); cursor: pointer; font-family: inherit;
    font-size: 13px; font-weight: 500; white-space: nowrap; transition: all .15s;
  }}
  .tab-btn:hover {{ color: var(--text); background: var(--bg3); }}
  .tab-btn.active {{
    color: var(--accent); background: var(--bg3);
    border-color: var(--border); border-bottom-color: var(--bg3);
  }}
  .tab-panel {{ display: none; padding: 24px 32px; }}
  .tab-panel.active {{ display: block; }}
  .section-title {{
    font-size: 16px; font-weight: 600; color: var(--text);
    margin-bottom: 16px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }}
  .chart-box {{
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 12px; padding: 16px; margin-bottom: 20px;
  }}
  .chart-box h3 {{ font-size: 13px; font-weight: 500; color: var(--muted); margin-bottom: 12px; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .grid-4 {{ display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 16px; }}
  @media(max-width:900px) {{ .grid-2,.grid-4 {{ grid-template-columns: 1fr; }} }}
  .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 20px; }}
  .stat-card {{
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }}
  .stat-card .title {{ font-size: 12px; color: var(--muted); margin-bottom: 8px; }}
  .stat-card .row {{ display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }}
  .stat-card .row span {{ color: var(--muted); }}
  .stat-card .row strong {{ color: var(--text); }}
  .pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;
  }}
  .pill.pass {{ background: rgba(79,204,142,.15); color: var(--green); border: 1px solid rgba(79,204,142,.3); }}
  .pill.fail {{ background: rgba(247,107,107,.15); color: var(--red);   border: 1px solid rgba(247,107,107,.3); }}
  .model-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .model-table th {{
    text-align: left; padding: 10px 14px;
    background: var(--bg3); color: var(--muted);
    font-size: 11px; text-transform: uppercase; letter-spacing: .5px;
    border-bottom: 1px solid var(--border);
  }}
  .model-table td {{ padding: 10px 14px; border-bottom: 1px solid var(--border); }}
  .model-table tr:last-child td {{ border-bottom: none; }}
  .best-banner {{
    background: linear-gradient(90deg, rgba(79,204,142,.15), rgba(79,204,142,.05));
    border: 1px solid rgba(79,204,142,.3); border-radius: 10px;
    padding: 16px 20px; margin-bottom: 20px;
    display: flex; align-items: center; gap: 12px;
  }}
  .best-banner .icon {{ font-size: 24px; }}
  .best-banner .label {{ font-size: 12px; color: var(--muted); }}
  .best-banner .model {{ font-size: 16px; font-weight: 700; color: var(--green); }}
  .forecast-table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 16px; }}
  .forecast-table th {{
    padding: 8px 12px; background: var(--bg3);
    color: var(--muted); font-size: 11px;
    text-transform: uppercase; letter-spacing: .5px;
    border-bottom: 1px solid var(--border);
  }}
  .forecast-table td {{ padding: 10px 12px; border-bottom: 1px solid var(--border); text-align: right; }}
  .forecast-table td:first-child {{ text-align: left; color: var(--muted); }}
  .plotly-chart {{ width: 100%; min-height: 340px; }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Series Temporales <span>· Diego Garcia Alvear</span></h1>
    <p>Métodos II · Prueba Final Ordinaria · Mayo 2026</p>
  </div>
  <span class="badge">SARIMA Analysis</span>
</header>

<div class="kpi-row" id="kpi-row"></div>

<div class="tabs-bar">
  <button class="tab-btn active" onclick="showTab('t1')">Serie Original</button>
  <button class="tab-btn" onclick="showTab('t2')">Descomposición</button>
  <button class="tab-btn" onclick="showTab('t3')">Transformaciones</button>
  <button class="tab-btn" onclick="showTab('t4')">ACF / PACF</button>
  <button class="tab-btn" onclick="showTab('t5')">Modelos ARIMA</button>
  <button class="tab-btn" onclick="showTab('t6')">Diagnóstico</button>
  <button class="tab-btn" onclick="showTab('t7')">Predicciones</button>
</div>

<div id="t1" class="tab-panel active">
  <div class="section-title">Serie Temporal Original</div>
  <div class="chart-box">
    <h3>Variable "Valor" — 500 observaciones semanales</h3>
    <div id="chart-orig" class="plotly-chart"></div>
  </div>
</div>

<div id="t2" class="tab-panel">
  <div class="section-title">Descomposición de la Serie (Aditiva, período = 7)</div>
  <div class="chart-box">
    <div id="chart-decomp" class="plotly-chart" style="min-height:600px"></div>
  </div>
</div>

<div id="t3" class="tab-panel">
  <div class="section-title">Transformaciones: Log y Diferenciación Logarítmica</div>
  <div class="grid-2">
    <div class="chart-box">
      <h3>Serie Log₁₀ (transformación Box-Cox signada)</h3>
      <div id="chart-log" class="plotly-chart"></div>
    </div>
    <div class="chart-box">
      <h3>Primera diferencia de Log₁₀ (tasa de variación semanal)</h3>
      <div id="chart-dlog" class="plotly-chart"></div>
    </div>
  </div>
</div>

<div id="t4" class="tab-panel">
  <div class="section-title">Funciones de Autocorrelación (ACF) y Autocorrelación Parcial (PACF)</div>
  <div class="grid-2">
    <div class="chart-box">
      <h3>ACF — Serie Original</h3>
      <div id="chart-acf-orig" class="plotly-chart"></div>
    </div>
    <div class="chart-box">
      <h3>PACF — Serie Original</h3>
      <div id="chart-pacf-orig" class="plotly-chart"></div>
    </div>
  </div>
  <div class="grid-2">
    <div class="chart-box">
      <h3>ACF — Serie dLog</h3>
      <div id="chart-acf-dlog" class="plotly-chart"></div>
    </div>
    <div class="chart-box">
      <h3>PACF — Serie dLog</h3>
      <div id="chart-pacf-dlog" class="plotly-chart"></div>
    </div>
  </div>
</div>

<div id="t5" class="tab-panel">
  <div class="section-title">Modelos ARIMA: Especificación y Ajuste</div>
  <div class="stat-grid" id="model-stats"></div>
  <div class="chart-box">
    <h3>Valores observados vs. ajustados</h3>
    <div id="chart-fitted" class="plotly-chart"></div>
  </div>
  <div class="chart-box" style="overflow-x:auto">
    <h3>Coeficientes estimados</h3>
    <table class="model-table" id="coef-table"></table>
  </div>
</div>

<div id="t6" class="tab-panel">
  <div class="section-title">Diagnóstico de Residuos</div>
  <div class="stat-grid" id="diag-stats"></div>
  <div class="grid-2">
    <div class="chart-box">
      <h3>Residuos — Modelo 1 (rojo)</h3>
      <div id="chart-resid1" class="plotly-chart"></div>
    </div>
    <div class="chart-box">
      <h3>Residuos — Modelo 2 (azul)</h3>
      <div id="chart-resid2" class="plotly-chart"></div>
    </div>
  </div>
</div>

<div id="t7" class="tab-panel">
  <div class="section-title">Predicciones: 4 periodos hacia adelante</div>
  <div id="best-banner"></div>
  <div class="chart-box">
    <h3>Últimas 50 observaciones + Forecast (IC 80% y 90%)</h3>
    <div id="chart-forecast" class="plotly-chart" style="min-height:420px"></div>
  </div>
  <div class="grid-2">
    <div class="chart-box">
      <h3>Tabla de predicciones — Modelo 1</h3>
      <table class="forecast-table" id="fc-table-1"></table>
    </div>
    <div class="chart-box">
      <h3>Tabla de predicciones — Modelo 2</h3>
      <table class="forecast-table" id="fc-table-2"></table>
    </div>
  </div>
</div>

<script>
const DATA = {data_json};

const LAYOUT_BASE = {{
  paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
  font: {{ family: 'Inter', color: '#e4e8f0', size: 12 }},
  margin: {{ t: 10, r: 20, b: 50, l: 70 }},
  xaxis: {{ gridcolor: '#2a3050', zerolinecolor: '#3a4060', title_font: {{size: 12}} }},
  yaxis: {{ gridcolor: '#2a3050', zerolinecolor: '#3a4060', title_font: {{size: 12}} }},
  showlegend: true,
  legend: {{ bgcolor: 'rgba(0,0,0,0)', font: {{size: 12}} }},
}};
const CFG = {{ responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['toImage','sendDataToCloud'] }};

function fmt(v) {{
  if (v === null || v === undefined || isNaN(v)) return '—';
  if (Math.abs(v) >= 1e6) return (v/1e6).toFixed(2) + 'M';
  if (Math.abs(v) >= 1e3) return (v/1e3).toFixed(1) + 'K';
  return v.toLocaleString('es-ES', {{maximumFractionDigits: 4}});
}}

function showTab(id) {{
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  const idx = ['t1','t2','t3','t4','t5','t6','t7'].indexOf(id);
  document.querySelectorAll('.tab-btn')[idx].classList.add('active');
  renderTab(id);
}}

const rendered = new Set();
function renderTab(id) {{
  if (rendered.has(id)) return;
  rendered.add(id);
  if (id === 't1') renderOrig();
  if (id === 't2') renderDecomp();
  if (id === 't3') renderTransforms();
  if (id === 't4') renderAcfPacf();
  if (id === 't5') renderModels();
  if (id === 't6') renderDiag();
  if (id === 't7') renderForecast();
}}

// KPI
(function() {{
  const k = DATA.kpi;
  const cards = [
    ['Observaciones', k.n],
    ['Media', fmt(k.mean)],
    ['Desv. Estándar', fmt(k.std)],
    ['Mediana', fmt(k.median)],
    ['Mínimo', fmt(k.min)],
    ['Máximo', fmt(k.max)],
  ];
  document.getElementById('kpi-row').innerHTML = cards.map(([l,v]) =>
    `<div class="kpi-card"><div class="label">${{l}}</div><div class="value">${{v}}</div></div>`
  ).join('');
}})();

// Tab 1 — Original
function renderOrig() {{
  const t = DATA.t_index, v = DATA.valores;
  Plotly.newPlot('chart-orig', [{{
    x: t, y: v, type: 'scatter', mode: 'lines',
    line: {{ color: '#4f8ef7', width: 1.2 }},
    name: 'Valor', hovertemplate: 'T=%{{x:.2f}}<br>Valor=%{{y:,.0f}}<extra></extra>'
  }}], {{...LAYOUT_BASE, yaxis: {{...LAYOUT_BASE.yaxis, title: 'Valor'}}, xaxis: {{...LAYOUT_BASE.xaxis, title: 'Período (semanas)'}}}}, CFG);
}}

// Tab 2 — Descomposición
function renderDecomp() {{
  const t = DATA.t_index;
  const d = DATA.decomp;
  const sub = (name, vals, col, row) => ({{
    x: t, y: vals, type: 'scatter', mode: 'lines',
    line: {{ color: col, width: 1.2 }}, name,
    xaxis: `x${{row>1?row:''}}`, yaxis: `y${{row>1?row:''}}`,
    hovertemplate: `${{name}}: %{{y:,.2f}}<extra></extra>`
  }});
  const traces = [
    sub('Observada', DATA.valores, '#4f8ef7', 1),
    sub('Tendencia', d.trend, '#f7c948', 2),
    sub('Estacional', d.seasonal, '#4fcc8e', 3),
    sub('Residuo', d.residual, '#f76b6b', 4),
  ];
  const layout = {{
    ...LAYOUT_BASE,
    grid: {{ rows: 4, columns: 1, pattern: 'independent', roworder: 'top to bottom' }},
    height: 680,
    margin: {{ t: 20, r: 20, b: 40, l: 70 }},
    xaxis4: {{ ...LAYOUT_BASE.xaxis, title: 'Período (semanas)' }},
    showlegend: false,
    annotations: [
      {{ text: 'Observada', xref:'paper', yref:'y domain', x:0, y:1, xanchor:'right', showarrow:false, font:{{size:11,color:'#4f8ef7'}} }},
      {{ text: 'Tendencia', xref:'paper', yref:'y2 domain', x:0, y:1, xanchor:'right', showarrow:false, font:{{size:11,color:'#f7c948'}} }},
      {{ text: 'Estacional', xref:'paper', yref:'y3 domain', x:0, y:1, xanchor:'right', showarrow:false, font:{{size:11,color:'#4fcc8e'}} }},
      {{ text: 'Residuo', xref:'paper', yref:'y4 domain', x:0, y:1, xanchor:'right', showarrow:false, font:{{size:11,color:'#f76b6b'}} }},
    ],
  }};
  Plotly.newPlot('chart-decomp', traces, layout, CFG);
}}

// Tab 3 — Transformaciones
function renderTransforms() {{
  const t = DATA.t_index;
  Plotly.newPlot('chart-log', [{{
    x: t, y: DATA.log_vals, type: 'scatter', mode: 'lines',
    line: {{ color: '#f76b6b', width: 1.2 }}, name: 'log₁₀(|Valor|)',
    hovertemplate: 'T=%{{x:.2f}}<br>log=%{{y:.4f}}<extra></extra>'
  }}], {{...LAYOUT_BASE, yaxis: {{...LAYOUT_BASE.yaxis, title: 'log₁₀'}}, xaxis: {{...LAYOUT_BASE.xaxis, title: 'Período'}}}}, CFG);

  Plotly.newPlot('chart-dlog', [{{
    x: t.slice(1), y: DATA.dlog_vals, type: 'scatter', mode: 'lines',
    line: {{ color: '#4f8ef7', width: 1.2 }}, name: 'Δlog₁₀',
    hovertemplate: 'T=%{{x:.2f}}<br>Δlog=%{{y:.4f}}<extra></extra>'
  }}], {{...LAYOUT_BASE, yaxis: {{...LAYOUT_BASE.yaxis, title: 'Δlog₁₀'}}, xaxis: {{...LAYOUT_BASE.xaxis, title: 'Período'}}}}, CFG);
}}

// Tab 4 — ACF/PACF
function barChart(divId, vals, ci, title, color) {{
  const lags = vals.map((_,i)=>i);
  Plotly.newPlot(divId, [
    {{ x: lags, y: vals, type: 'bar', marker: {{ color: color }}, name: title,
       hovertemplate: 'Lag=%{{x}}<br>r=%{{y:.4f}}<extra></extra>' }},
    {{ x: [0, lags.length-1], y: [ci, ci], type: 'scatter', mode: 'lines',
       line: {{ color: '#f7c948', dash: 'dash', width: 1 }}, showlegend: false }},
    {{ x: [0, lags.length-1], y: [-ci, -ci], type: 'scatter', mode: 'lines',
       line: {{ color: '#f7c948', dash: 'dash', width: 1 }}, showlegend: false }},
  ], {{...LAYOUT_BASE, bargap: 0.1, yaxis: {{...LAYOUT_BASE.yaxis, title: 'Autocorrelación', range:[-1,1]}},
       xaxis: {{...LAYOUT_BASE.xaxis, title: 'Lag'}}}}, CFG);
}}
function renderAcfPacf() {{
  const a = DATA.acf_pacf;
  barChart('chart-acf-orig',  a.acf_orig,  a.ci_band, 'ACF',  '#4f8ef7');
  barChart('chart-pacf-orig', a.pacf_orig, a.ci_band, 'PACF', '#4fcc8e');
  barChart('chart-acf-dlog',  a.acf_dlog,  a.ci_band, 'ACF',  '#f76b6b');
  barChart('chart-pacf-dlog', a.pacf_dlog, a.ci_band, 'PACF', '#f7c948');
}}

// Tab 5 — Modelos
function renderModels() {{
  const m = DATA.models;
  const statHtml = (title, aic, bic, llf, color) => `
    <div class="stat-card">
      <div class="title" style="color:${{color}}">${{title}}</div>
      <div class="row"><span>AIC</span><strong>${{aic.toFixed(2)}}</strong></div>
      <div class="row"><span>BIC</span><strong>${{bic.toFixed(2)}}</strong></div>
      <div class="row"><span>Log-Lik</span><strong>${{llf.toFixed(2)}}</strong></div>
    </div>`;
  document.getElementById('model-stats').innerHTML =
    statHtml(m.m1.name, m.m1.aic, m.m1.bic, m.m1.llf, '#f76b6b') +
    statHtml(m.m2.name, m.m2.aic, m.m2.bic, m.m2.llf, '#4f8ef7');

  const t = DATA.t_index;
  Plotly.newPlot('chart-fitted', [
    {{ x:t, y:DATA.valores, mode:'lines', line:{{color:'#7a85a0',width:1}}, name:'Observada' }},
    {{ x:t, y:m.fitted1, mode:'lines', line:{{color:'#f76b6b',width:1.5}}, name:'Ajuste M1' }},
    {{ x:t, y:m.fitted2, mode:'lines', line:{{color:'#4f8ef7',width:1.5}}, name:'Ajuste M2' }},
  ], {{...LAYOUT_BASE, xaxis:{{...LAYOUT_BASE.xaxis, title:'Período'}}, yaxis:{{...LAYOUT_BASE.yaxis, title:'Valor'}}}}, CFG);

  // Coeficientes table
  const allParams = new Set([...Object.keys(m.m1.params), ...Object.keys(m.m2.params)]);
  let rows = `<thead><tr><th>Parámetro</th><th style="color:#f76b6b">Modelo 1</th><th>p-valor M1</th><th style="color:#4f8ef7">Modelo 2</th><th>p-valor M2</th></tr></thead><tbody>`;
  allParams.forEach(k => {{
    const v1 = m.m1.params[k], p1 = m.m1.pvalues[k];
    const v2 = m.m2.params[k], p2 = m.m2.pvalues[k];
    rows += `<tr><td>${{k}}</td><td>${{v1!==undefined ? v1.toFixed(4) : '—'}}</td><td>${{p1!==undefined ? p1.toFixed(4) : '—'}}</td><td>${{v2!==undefined ? v2.toFixed(4) : '—'}}</td><td>${{p2!==undefined ? p2.toFixed(4) : '—'}}</td></tr>`;
  }});
  document.getElementById('coef-table').innerHTML = rows + '</tbody>';
}}

// Tab 6 — Diagnóstico
function renderDiag() {{
  const m = DATA.models, adf = DATA.adf;
  const pass = (p, thr) => p < thr;
  const pill = (p, thr, label) => `<span class="pill ${{pass(p,thr)?'pass':'fail'}}">${{pass(p,thr)?'✓':'✗'}} ${{label}} p=${{p.toFixed(4)}}</span>`;

  document.getElementById('diag-stats').innerHTML = `
    <div class="stat-card">
      <div class="title">Test ADF (Dickey-Fuller Aumentado)</div>
      <div class="row"><span>Serie original</span><strong>${{adf.orig.stat.toFixed(4)}}</strong></div>
      <div class="row" style="padding-bottom:8px"><span>p-valor</span>${{pill(adf.orig.pval, 0.1, 'α=0.10')}}</div>
      <div class="row"><span>Serie dLog</span><strong>${{adf.dlog.stat.toFixed(4)}}</strong></div>
      <div class="row"><span>p-valor</span>${{pill(adf.dlog.pval, 0.1, 'α=0.10')}}</div>
    </div>
    <div class="stat-card">
      <div class="title">Test Ljung-Box (lag=10)</div>
      <div class="row"><span>Modelo 1 p-valor</span>${{pill(m.lb_pval1, 0.1, 'H₀ res. incorrelados')}}</div>
      <div class="row" style="padding-top:8px"><span>Modelo 2 p-valor</span>${{pill(m.lb_pval2, 0.1, 'H₀ res. incorrelados')}}</div>
      <div style="margin-top:12px;font-size:11px;color:var(--muted)">
        Si p &gt; 0.10 → residuos incorrelados (buen ajuste)
      </div>
    </div>`;

  const t = DATA.t_index;
  Plotly.newPlot('chart-resid1', [{{
    x:t, y:m.resid1, mode:'lines', line:{{color:'#f76b6b',width:1}}, name:'Residuos M1'
  }}], {{...LAYOUT_BASE, xaxis:{{...LAYOUT_BASE.xaxis,title:'Período'}}, yaxis:{{...LAYOUT_BASE.yaxis,title:'Residuo'}}}}, CFG);

  Plotly.newPlot('chart-resid2', [{{
    x:t, y:m.resid2, mode:'lines', line:{{color:'#4f8ef7',width:1}}, name:'Residuos M2'
  }}], {{...LAYOUT_BASE, xaxis:{{...LAYOUT_BASE.xaxis,title:'Período'}}, yaxis:{{...LAYOUT_BASE.yaxis,title:'Residuo'}}}}, CFG);
}}

// Tab 7 — Forecast
function renderForecast() {{
  const m = DATA.models, fc = DATA.forecast;
  const best = m.m1.aic < m.m2.aic ? 'Modelo 1' : 'Modelo 2';
  const bestAIC = Math.min(m.m1.aic, m.m2.aic);
  document.getElementById('best-banner').innerHTML = `
    <div class="best-banner">
      <span class="icon">★</span>
      <div><div class="label">Mejor modelo según AIC</div><div class="model">${{best}} — AIC = ${{bestAIC.toFixed(2)}}</div></div>
    </div>`;

  const t = DATA.t_index;
  const tail = 50;
  const tHist = t.slice(-tail);
  const vHist = DATA.valores.slice(-tail);
  const lastT = t[t.length - 1];
  const fcT = [1,2,3,4].map(i => lastT + i/7);

  const m1ci80 = fc.m1.ci80, m1ci90 = fc.m1.ci90;
  const m2ci80 = fc.m2.ci80, m2ci90 = fc.m2.ci90;

  Plotly.newPlot('chart-forecast', [
    {{ x:tHist, y:vHist, mode:'lines', line:{{color:'#7a85a0',width:1.5}}, name:'Observada (últimas 50)' }},
    {{ x:fcT, y:fc.m1.mean, mode:'lines+markers', line:{{color:'#f76b6b',width:2,dash:'dot'}}, marker:{{size:6}}, name:'Forecast M1' }},
    {{ x:[...fcT,...[...fcT].reverse()], y:[...m1ci80.map(r=>r[1]),...[...m1ci80.map(r=>r[0])].reverse()],
       fill:'toself', fillcolor:'rgba(247,107,107,0.15)', line:{{color:'transparent'}}, name:'IC 80% M1', showlegend:true }},
    {{ x:[...fcT,...[...fcT].reverse()], y:[...m1ci90.map(r=>r[1]),...[...m1ci90.map(r=>r[0])].reverse()],
       fill:'toself', fillcolor:'rgba(247,107,107,0.08)', line:{{color:'transparent'}}, name:'IC 90% M1', showlegend:true }},
    {{ x:fcT, y:fc.m2.mean, mode:'lines+markers', line:{{color:'#4f8ef7',width:2,dash:'dot'}}, marker:{{size:6}}, name:'Forecast M2' }},
    {{ x:[...fcT,...[...fcT].reverse()], y:[...m2ci80.map(r=>r[1]),...[...m2ci80.map(r=>r[0])].reverse()],
       fill:'toself', fillcolor:'rgba(79,142,247,0.15)', line:{{color:'transparent'}}, name:'IC 80% M2', showlegend:true }},
    {{ x:[...fcT,...[...fcT].reverse()], y:[...m2ci90.map(r=>r[1]),...[...m2ci90.map(r=>r[0])].reverse()],
       fill:'toself', fillcolor:'rgba(79,142,247,0.08)', line:{{color:'transparent'}}, name:'IC 90% M2', showlegend:true }},
  ], {{...LAYOUT_BASE, xaxis:{{...LAYOUT_BASE.xaxis,title:'Período (semanas)'}}, yaxis:{{...LAYOUT_BASE.yaxis,title:'Valor'}}, height:420}}, CFG);

  const fcTableHtml = (fcData, colClass) => {{
    const hdr = `<thead><tr><th>Período</th><th>Predicción</th><th>IC80 Inf</th><th>IC80 Sup</th><th>IC90 Inf</th><th>IC90 Sup</th></tr></thead>`;
    const rows = fcData.mean.map((v,i) => `<tr>
      <td>T+${{i+1}}</td>
      <td style="color:${{colClass}}">${{fmt(v)}}</td>
      <td>${{fmt(fcData.ci80[i][0])}}</td><td>${{fmt(fcData.ci80[i][1])}}</td>
      <td>${{fmt(fcData.ci90[i][0])}}</td><td>${{fmt(fcData.ci90[i][1])}}</td>
    </tr>`).join('');
    return hdr + '<tbody>' + rows + '</tbody>';
  }};
  document.getElementById('fc-table-1').innerHTML = fcTableHtml(fc.m1, '#f76b6b');
  document.getElementById('fc-table-2').innerHTML = fcTableHtml(fc.m2, '#4f8ef7');
}}

// Render Tab 1 on load
renderTab('t1');
</script>
</body>
</html>"""


# ── GENERAR HTML ──────────────────────────────────────────────────────────────
DATA_JSON = json.dumps(data, ensure_ascii=False, allow_nan=False)
html = build_html(DATA_JSON)
with open("index.html", "w", encoding="utf-8") as fh:
    fh.write(html)
print("index.html generado correctamente.")
