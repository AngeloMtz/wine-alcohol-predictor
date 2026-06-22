from flask import Flask, request, render_template
import joblib
import pandas as pd
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

# ── Cargar pipeline al iniciar ────────────────────────────────────────────────
pipeline = joblib.load('pipeline_alcohol.pkl')
app.logger.debug('Pipeline cargado correctamente.')

# Orden exacto de features que espera el pipeline (mismo del entrenamiento)
FEATURES = ['density', 'residual sugar', 'quality', 'fixed acidity', 'pH']

# Rangos de referencia para validación (basados en el dataset)
RANGOS = {
    'density'       : (0.987, 1.039, 'g/mL'),
    'residual sugar': (0.6,   65.8,  'g/L'),
    'quality'       : (3,     9,     'puntos'),
    'fixed acidity' : (3.8,   14.2,  'g/L'),
    'pH'            : (2.72,  3.82,  ''),
}

# ── Ruta principal ────────────────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

# ── Ruta de predicción ────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        density        = float(request.form['density'])
        residual_sugar = float(request.form['residual_sugar'])
        quality        = int(request.form['quality'])
        fixed_acidity  = float(request.form['fixed_acidity'])
        ph             = float(request.form['ph'])

        # Validaciones de rango
        errores = []
        vals = {
            'density': density, 'residual sugar': residual_sugar,
            'quality': quality, 'fixed acidity': fixed_acidity, 'pH': ph
        }
        for campo, val in vals.items():
            mn, mx, _ = RANGOS[campo]
            if not (mn <= val <= mx):
                errores.append(f'{campo}: valor {val} fuera del rango esperado [{mn}, {mx}]')

        if errores:
            return render_template('index.html',
                                   error=' | '.join(errores),
                                   valores=request.form)

        # Construir DataFrame con orden correcto
        df_input = pd.DataFrame([[density, residual_sugar, quality,
                                   fixed_acidity, ph]], columns=FEATURES)

        app.logger.debug(f'Input: {df_input.to_dict()}')

        # Predecir (el pipeline aplica imputación + escalado internamente)
        pred = pipeline.predict(df_input)[0]
        pred = round(float(pred), 2)

        app.logger.debug(f'Predicción: {pred}% ABV')

        return render_template('index.html',
                               prediccion=pred,
                               valores=request.form)

    except ValueError:
        return render_template('index.html',
                               error='Por favor ingresa valores numéricos válidos en todos los campos.',
                               valores=request.form)
    except Exception as e:
        app.logger.error(f'Error: {str(e)}')
        return render_template('index.html',
                               error=f'Error interno: {str(e)}',
                               valores=request.form)

if __name__ == '__main__':
    app.run(debug=True)
