import os
import re
import json
import unicodedata
import sys
import random
import mysql.connector
import numpy as np
import tensorflow as tf

from gensim import utils
from keras.models import load_model
from flask import Flask, jsonify, request

app = Flask(__name__)

#connect to database
db = mysql.connector.connect(user="root", password="12345678", database="glove-300")
cursor = db.cursor(buffered=True)

# load cnn model
model = load_model("./CNN+GRU2_1_update_05.h5")
model._make_predict_function()

def getPrediction(doc):
    vec = toSentenceEmbd(doc)
    print (vec)
    vec = vec.reshape((1, 300, 1))
    prediction = model.predict([vec])[0]
    print (prediction)
    argmax = np.argmax(prediction)
    return [argmax+1]

def getWordEmbedding(word, cursor):
#     word = word.replace("'", "''")
    sql = """select vec from term where term like %s"""
    cursor.execute(sql, (str(word),))
    data = cursor.fetchall()
    if len(data) > 0:
        decoded_vec = json.JSONDecoder().decode(data[0][0])
        vec = np.asarray(decoded_vec, dtype=np.float32)
        return True, vec
    else:
        return False, data

def myTokenizer(content, lower=True):
    raw = content.split(' ')
    remover = re.compile("[^a-zA-Z-]")

    token = []

    for i in raw:
        term = remover.sub('', i)
        if lower == True:
            term = term.lower()
        token.append(term)
    tokenized = filter(None, token)

    return tokenized

def toSentenceEmbd(string):
    string = string.replace('\n', '')
    string = np.array(list(myTokenizer(string)))

    begin = True
    for word in string:
        stat, vec = getWordEmbedding(word, cursor)
        if not stat:
            continue
        if begin:
            begin = False
            feature = vec
        else:
            feature += vec
            # feature = np.concatenate([feature, vec])

    feature = feature/np.linalg.norm(feature)
    feature = np.array(feature)

    return feature

def getAnswer(dictionary):
    dictionary = str(dictionary)
    with open('messagesnw.json', 'r') as data_file:
        data = json.load(data_file, strict=False)
    return random.choice(data[dictionary])

@app.route("/predict", methods=["POST"])
def ApiCall():
    text = request.values['question']
    sys.stdout.write(text+'\n')

    id = getPrediction(text)
    answer = getAnswer(id[0])
    return jsonify(answer=answer)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
