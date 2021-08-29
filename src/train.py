"""
this module contains functions to train a NER model using spacy.
"""
import random
import spacy

TRAIN_DATA = [('what is the price of polo?', {'entities': [(21, 25, 'PrdName')]}),
              ('what is the price of ball?', {'entities': [(21, 25, 'PrdName')]}),
              ('what is the price of jegging?', {'entities': [(21, 28, 'PrdName')]}),
              ('what is the price of t-shirt?', {'entities': [(21, 28, 'PrdName')]}),
              ('what is the price of jeans?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of bat?', {'entities': [(21, 24, 'PrdName')]}),
              ('what is the price of shirt?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of bag?', {'entities': [(21, 24, 'PrdName')]}),
              ('what is the price of cup?', {'entities': [(21, 24, 'PrdName')]}),
              ('what is the price of jug?', {'entities': [(21, 24, 'PrdName')]}),
              ('what is the price of plate?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of glass?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of moniter?', {'entities': [(21, 28, 'PrdName')]}),
              ('what is the price of desktop?', {'entities': [(21, 28, 'PrdName')]}),
              ('what is the price of bottle?', {'entities': [(21, 27, 'PrdName')]}),
              ('what is the price of mouse?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of keyboad?', {'entities': [(21, 28, 'PrdName')]}),
              ('what is the price of chair?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of table?', {'entities': [(21, 26, 'PrdName')]}),
              ('what is the price of watch?', {'entities': [(21, 26, 'PrdName')]})]


def training(training_data, iterations):
    """
    this funtion trains a custom NER model on the given training data for given iterations
    :param training_data: list
    :param iterations: int
    :return:
    """
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    # add labels
    for _, annotations in training_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in training_data:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    return nlp


def save_model(model):
    """
    this functions saves the given ner model to disk
    :param model: spacy ner model
    :return:
    """
    model.to_disk("../model/custom_ner_model")


def test_model():
    """
    Its quick test for NER model
    :return:
    """
    test_text = "what is the price of jug?"
    model = spacy.load("../model/custom_ner_model")
    doc = model(test_text)
    for ent in doc.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)


def convert_data(data):
    """
    This function converts the raw training data into Spacy compatible format
    :param data: list
    :return: list
    """
    training_data = []
    for record in data:
        entities = record.get('entities')
        converted_entities = []
        for entity in entities:
            converted_entities.append((entity.get('start_offset'), entity.get('end_offset'),
                                       entity.get('entity_name')))

        training_data.append((record.get('text'), {'entities': converted_entities}))
    return training_data


if __name__ == '__main__':
    data = {
        "training_data": [
            {
                "text": "what is the price of polo?",
                "entities": [
                    {
                        "start_offset": 21,
                        "end_offset": 25,
                        "entity_name": "PrdName"
                    }
                ]
            },
            {
                "text": "what is the price of ball?",
                "entities": [
                    {
                        "start_offset": 21,
                        "end_offset": 25,
                        "entity_name": "PrdName"
                    }
                ]
            },
            {
                "text": "what is the price of jegging?",
                "entities": [
                    {
                        "start_offset": 21,
                        "end_offset": 28,
                        "entity_name": "PrdName"
                    }
                ]
            },
            {
                "text": "what is the price of t-shirt?",
                "entities": [
                    {
                        "start_offset": 21,
                        "end_offset": 28,
                        "entity_name": "PrdName"
                    }
                ]
            }
        ]
    }
    convert_data(data.get('training_data'))
