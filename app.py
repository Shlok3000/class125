from flask import Flask, jsonify, request
from model import getPrediction

app = Flask(__name__)

@app.route('/')
def getData():
    return('Hello World')

@app.route('/predict-digit', methods=["POST"])
def addData():
    digitImage = request.files.get("digit")
    prediction = getPrediction(digitImage)
    return jsonify({
        "prediction": prediction
    }),200

if(__name__=="__main__"):
    app.run(debug=True)