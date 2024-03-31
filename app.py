from flask import Flask,render_template,jsonify,app,url_for,request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data =request.json