
from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from classifier_helper import *
import base64

app = Flask(__name__)
cors = CORS(app)


@app.route('/predict/knn', methods=['POST'])
def predict():
    data = request.get_json()
    response = knn_helper(1)
    return {"id": "knn", "response": response}

@app.route('/predict/uniform-MAP', methods=['POST'])
def predict_uniform_MAP():
    # data = request.get_json()
    # response = None
    # if data["type"] == "knn":
    #     # knn_number = data.get("knn_number", 1)  # Default to 1 if 'knn_number' is not provided
    #     response = knn_helper(1)
    response = MAP_uniform_helper()
    return {"id": "uniform-MAP", "response": response}

@app.route('/predict/non-uniform-MAP', methods=['POST'])
def predict_non_uniform_MAP():
    response = MAP_non_uniform_helper()
    return {"id": "non-uniform-MAP", "response": response}

@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    # data = request.get_json()
    response = svm_helper()
    return {"id": "svm", "response": response} 

  
if __name__ == '__main__':
    app.run(debug=True)  

