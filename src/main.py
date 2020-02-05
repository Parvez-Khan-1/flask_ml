from src import constant
from src import predict
from src import train
from flask import Flask, request, jsonify
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)


@app.route("/health_check", methods=['GET'])
def hello():
    """
        This is the Health Check API
        Call this api to check if a micro-service is up and alive
        ---
        tags:
          - Health Check
        responses:
            200:
                description: The service is up and alive
        """
    return jsonify(constant.TEST_MESSAGE)


@app.route("/predict", methods=['POST'])
def make_prediction():
    """
    Predict
    Invoke this API To Extract Named Entities from the input text
    ---
    tags:
        - Predict Named Entities
    """
    parameters = request.get_json(force=True, silent=True)
    print(parameters)
    text = parameters.get('text')
    prediction = predict.predict_named_entities(text)
    if prediction is not None:
        return jsonify(prediction)
    else:
        return jsonify("Sorry Machine Learning Wont be able to identify Named Entities in given text"), 500


@app.route("/predict_pretrained", methods=['POST'])
def make_pretrained_prediction():
    """
    Predict From Pre-trained Model
    Invoke this API To Extract Named Entities from Spacy's pretrained NER Model
    ---
    tags:
        - Predict Named Entities From Pre-trained SpaCy Model
    """
    parameters = request.get_json(force=True, silent=True)
    print(parameters)
    text = parameters.get('text')
    prediction = predict.predict_named_entities_from_pre_trained_model(text)
    if prediction is not None:
        return jsonify(prediction)
    else:
        return jsonify("Sorry Machine Learning Wont be able to identify Named Entities in given text"), 500


@app.route("/train", methods=['POST'])
def train_model():
    """
    Model Training
    Invoke this API To Train a Named Entity Recognition Model
    ---
    tags:
      - Train a Named Entity Recognition Model
    """
    parameters = request.get_json(force=True, silent=True)
    raw_data = parameters.get('training_data', None)
    iterations = parameters.get('epochs', None)

    if raw_data is None:
        return jsonify("Please provide valid training data")

    training_data = train.convert_data(raw_data)

    # Train a NER Model
    custom_train_model = train.training(training_data, iterations)

    # Save a Model
    train.save_model(custom_train_model)
    return jsonify({"message": constant.TRAINING_MESSAGE, "evaluation_metrics": ["You will soon receive the evaluation metrics."]})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
