# Series Temporales — Diego Garcia Alvear
**Métodos II · Prueba Final Ordinaria · Mayo 2026**

Dashboard interactivo de análisis SARIMA sobre serie temporal semanal (500 observaciones).

## Contenido del dashboard

| Pestaña | Análisis |
|---------|----------|
| Serie Original | Visualización interactiva de los 500 datos |
| Descomposición | Tendencia, componente estacional y residuo (período 7) |
| Transformaciones | Serie log₁₀ y primera diferencia (tasa de variación) |
| ACF / PACF | Funciones de autocorrelación (lag máx. 100) |
| Modelos ARIMA | SARIMA(1,1,0)(0,1,1)₇ vs SARIMA(2,1,0)(2,1,0)₇ |
| Diagnóstico | Test ADF, Ljung-Box y análisis de residuos |
| Predicciones | Forecast 4 períodos con IC 80% y 90% |

## Deploy en Vercel

### 1. Inicializar repositorio Git

```bash
git init
git add index.html vercel.json package.json README.md generate_analysis.py
git commit -m "Add time series dashboard"
```

### 2. Subir a GitHub

```bash
git remote add origin https://github.com/TU_USUARIO/series-temporales-dgao.git
git branch -M main
git push -u origin main
```

### 3. Deploy en Vercel

1. Entra en [vercel.com](https://vercel.com) e inicia sesión.
2. Haz clic en **"New Project"** e importa el repositorio.
3. En la configuración, deja el **Framework Preset** en `Other`.
4. Haz clic en **Deploy**.

La URL pública quedará disponible automáticamente.

## Regenerar el dashboard

Si modificas los datos o el análisis, regenera el `index.html` con:

```bash
python generate_analysis.py
```
