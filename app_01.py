import json             # bekerja dengan data berformat JSON
import pickle           # membaca file model Machine Learning

# Flask -- membuat aplikasi web
# request -- menerima data dari browser
# jsonify  -- mengubah hasil python menjadi format json
# render_template -- menampilkan halaman HTML
from flask import Flask,request,app,jsonify,url_for,render_template

import numpy as np      # pengolahan array
import pandas as pd     # pengolahan dataframe

app=Flask(__name__)     # object aplikasi flask bernama flask
                        # memiliki endpoint route

import os

APP_SECRET = os.getenv("APP_SECRET")

MODEL_FILE = os.getenv("MODEL_FILE", "regmodel.pkl")
SCALER_FILE = os.getenv("SCALER_FILE", "scaling.pkl")

scalar=pickle.load(open(SCALER_FILE,'rb'))        # rb is read binary
regmodel=pickle.load(open(MODEL_FILE,'rb'))     # rb is read binary

## Load the model
# scalar=pickle.load(open('scaling.pkl','rb'))        # rb is read binary
# regmodel=pickle.load(open('regmodel.pkl','rb'))     # rb is read binary

# Endpoint root (home)
@app.route('/')
def home():
    return render_template('home.html')

# Endpoint prediksi harga rumah
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    # data.values is dataframe, convert into list, convert into array

    # standard scalar (standardization/normalization)
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=regmodel.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

@app.route('/health')
def health():
    return "OK", 200

'''
if __name__=="__main__":
    app.run(debug=True)
'''
         
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)