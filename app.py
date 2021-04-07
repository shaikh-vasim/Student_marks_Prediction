import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
import math
import pandas as pd


app = Flask(__name__)
model = joblib.load('student_mark_predictor_ml')


@app.route('/')
def home():

    first = request.args.get('first')

    return render_template('index.html', first=first)


@app.route('/pred')
def pred():
    return render_template('modal.html')


@app.route('/predict', methods=['POST'])
def predict():
    fname = request.form.get('fname')
    number = request.form.get('number')

    input_features = [int(number)]

    final_features = np.array(input_features)
    output = model.predict([final_features]).round(2)
    pred_out = output

    # return render_template('modal.html', prediction_text="{}You will get  {} %  marks, when you do study hours per day".format(int(output[0])))

    return render_template('/modal.html', pred_out=pred_out, fname=fname, number=number, done="prediction is done", show="Show Output")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
    # app.run(debug=True)
