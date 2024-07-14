
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

data = pd.read_csv('Bangalorehousingprice.csv')
delhid = pd.read_csv('Delhihousingprice.csv')


model = joblib.load('XGBRegressor.joblib')
modeld = joblib.load('LinearRegressor.joblib')

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


@app.route("/delhi")
def delhi():
  locality = sorted(delhid.Locality.unique())
  locality.insert(0, ' ')
  furnishing = sorted(delhid.Furnishing.unique())
  furnishing.insert(0, ' ')
  return render_template('delhi.html', localities=locality, furnishies=furnishing)


@app.route('/predictdelhi', methods=['POST'])
def predictdelhi():
  locality = request.form.get('locality')
  furnishing = request.form.get('furnishing')
  bathroom = request.form.get('bathroom')
  bhk = request.form.get('bhk')
  area = request.form.get('area')
  print(locality, furnishing, bathroom, bhk, area)
  input = pd.DataFrame([[area, bhk, bathroom, furnishing, locality]], columns=['Area', 'BHK', 'Bathroom', 'Furnishing', 'Locality'])
  prediction = modeld.predict(input)[0]
  return str(np.round(prediction), 1)
  



if __name__ == "__main__":
  app.run(debug=True, host="localhost", port=5002
)