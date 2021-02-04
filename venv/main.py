import pickle

import nltk
nltk.data.path.append("viveknlpflask/nltk_data")
nltk.data.path.append("viveknlpflask-303807/nltk_data")

from nltk import word_tokenize



from flask import Flask, render_template, request

# Using Pickle file for now later on should move to cloud buckets
import requests
from io import BytesIO

foodtech_pickle = "https://github.com/viveknest/FlaskProject/blob/master/venv/foodtech.pickle?raw=true"
foodfeaturespickle = "https://github.com/viveknest/FlaskProject/blob/master/venv/foodtechfeatures.pickle?raw=true"

# # For google cloud
# import os
# from google.appengine.api import app_identity
#
# # import file on cloud



app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template("main.html")

classfile = BytesIO(requests.get(foodtech_pickle).content)
featfile = BytesIO(requests.get(foodfeaturespickle).content)

# classifier_f = open(foodtech_pickle, "rb")
kick_classifier = pickle.load(classfile)
# classifier_f.close()

# features_f = open(foodtechfeatures.pickle, "rb")
kick_features = pickle.load(featfile)
# features_f.close()

def kick_find_features(document):
    words = set(document)
    features = {}
    for w in kick_features:
        features[w] = (w in words)

    return features

@app.route("/classify", methods=["POST"])
def classify():
    if request.method == "POST":
        input_text = request.form.get("input_text")
        user_input = input_text
        input_tokenize = word_tokenize(user_input.lower())
        pred_class = kick_classifier.classify(kick_find_features(input_tokenize))
        pred_prob = kick_classifier.prob_classify(kick_find_features(input_tokenize)).prob(pred_class) * 100

        classification =  '%s %.2f' % (pred_class, pred_prob)
    return render_template("classify.html", classification=classification)


if __name__ == "__main__":
    app.run()
