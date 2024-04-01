from flask import Flask,render_template,jsonify,app,url_for,request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaledmodel = pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    data =request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    transformed_data = scaledmodel.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(transformed_data)
    print(output[0])
    return jsonify(output[0])


if __name__ == "__main__":
    app.run(debug=True)
    
