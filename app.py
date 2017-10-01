"""
Flask Serving
This file is a sample flask app that can be used to test your model with an REST API.
This app does the following:
    - Create a test sample from a normal distribution with zero mean and variance one
    - Returns the regression value
Additional configuration:
    - You can also choose the checkpoint file name to use as a request parameter
    - Parameter name: ckp
    - It is loaded from /model

POST req:
    parameter:
        - ckp, optional, load a specific chekcpoint from /model

"""
import os
import torch
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename
from regression_model import LinearRegression

MODEL_PATH = '/input'
print('Loading model from path: %s' % MODEL_PATH)

app = Flask('Linear-Regression')

# Return an Image
@app.route('/<path:path>', methods=['POST'])
def geneator_handler(path):
    # Get ckp
    checkpoint = request.form.get("ckp") or "regression_4_degree_polynomial.pth"
    checkpoint = os.path.join(MODEL_PATH, checkpoint)
    # Preprocess, Build and Evaluate
    Model = LinearRegression(ckp=checkpoint)
    Model.build_model()
    return Model.evaluate()

if __name__ == '__main__':
    app.run(host='0.0.0.0')
