from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

app = Flask(__name__)

# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

# Cargar el modelo entrenado y el escalador
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

app.logger.debug('Modelo y escalador cargados correctamente.')

@app.route('/')
def home():
    return render_template('formulario.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        MINOR_AXIS = float(request.form['MINOR_AXIS'])
        SOLIDITY = float(request.form['SOLIDITY'])
        
        # Escalar los datos de entrada automáticamente
        input_data = pd.DataFrame([[MINOR_AXIS, SOLIDITY]], columns=['MINOR_AXIS', 'SOLIDITY'])
        input_data_scaled = scaler.transform(input_data)
        
        # Realizar predicciones
        prediction = model.predict(input_data_scaled)
        app.logger.debug(f'Predicción: {prediction[0]}')
        
        # Devolver las predicciones como respuesta JSON
        return jsonify({'categoria': prediction[0]})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
