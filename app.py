
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

data = pd.read_csv('Bangalorehousingprice.csv')

model = joblib.load('XGBRegressor.joblib')

@app.route('/')
def index():
  location = sorted(data.location.unique())
  location.insert(0, ' ')
  return render_template('index.html', locations=location)

@app.route('/predict', methods=['POST'])
def predict():
  location = request.form.get('location')
  sqft = request.form.get('sqft')
  bathroom = request.form.get('bathroom')
  bhk = request.form.get('bhk')
  pricesqft = request.form.get('pricesqft')
  print(location, sqft, bathroom, bhk, pricesqft)
  input = pd.DataFrame([[location, sqft, bathroom, bhk, pricesqft]], columns=['location', 'total_sqft', 'bath', 'bhk', 'price_per_sqft'])
  prediction = model.predict(input)[0] * 10000
  return str(np.round(prediction, 2))


if __name__ == "__main__":
  app.run(debug=True, host="localhost", port=5002
)