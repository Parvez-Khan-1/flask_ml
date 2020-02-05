import spacy

# Load Pre trained NER Model
pre_trained_model = spacy.load("en_core_web_sm")


def predict_named_entities(text):
    # Load Custom NER Model
    model = spacy.load("../model/custom_ner_model")
    doc = model(text)
    prediction = []
    for ent in doc.ents:
        prediction.append({
            'text': ent.text,
            'start_char': ent.start_char,
            'end_char': ent.end_char,
            'label': ent.label_
        })
    return prediction


def predict_named_entities_from_pre_trained_model(text):
    doc = pre_trained_model(text)
    prediction = []
    for ent in doc.ents:
        prediction.append({
            'text': ent.text,
            'start_char': ent.start_char,
            'end_char': ent.end_char,
            'label': ent.label_
        })
    return prediction
