# Core Packages
import tkinter as tk
from tkinter import *
from tkinter import ttk
import numpy as np
from keras.models import load_model


#
import re
from textblob import TextBlob
import numpy as np
from textblob import Word
from textblob.classifiers import NaiveBayesClassifier
import nltk
from nltk.probability import FreqDist
import random
import numpy as np
from textblob import TextBlob
reconstructed_model = load_model("keras1")
reconstructed_model.summary()
# Structure and Layout
window = Tk()
window.title("Summaryzer GUI")
window.geometry("700x400")
window.config()
style = ttk.Style(window)
style.configure('lefttab.TNotebook', tabposition='wn',)

# TAB LAYOUT
tab_control = ttk.Notebook(window, style='lefttab.TNotebook')


def classify_review():
    # import pickle
    # f = open('svm.sav', 'rb')
    # classifier = pickle.load(f)
    raw_text = str(entry.get('1.0', tk.END))
# start
    tokens = nltk.word_tokenize(raw_text)
    freq = nltk.FreqDist(tokens)
    tages = nltk.pos_tag(tokens)

    # print(freq.most_common(10))

    #
    bigrams = nltk.bigrams(raw_text.split(" "))
    bigram_freq = nltk.FreqDist(list(bigrams))
    sorted_bigram_freq = bigram_freq.most_common()
    print(sorted_bigram_freq[20:80])

    #
    trigrams = nltk.trigrams(raw_text.split(" "))
    trigram_freq = nltk.FreqDist(list(trigrams))
    sorted_trigram_freq = trigram_freq.most_common()
    print(sorted_trigram_freq[10:20])

    #
    wnl = nltk.WordNetLemmatizer()

    def verb_checker(pair):
        if pair[1].startswith("V"):

            return Word(pair[0]).lemmatize("v")
            # return wnl.lemmatize(pair[0])
        else:
            return pair[0]

    new_tokens = [verb_checker(item) for item in tages]
    print(new_tokens[:5])
    new_tokens_tages = nltk.pos_tag(new_tokens)
    dic_of_pos = {"VB": [], "JJ": [], "RB": [], "NN": []}
    for token in set(new_tokens):
        pair = nltk.pos_tag([token])[0]
        if pair[1].startswith("VB"):
            dic_of_pos["VB"].append(pair[0])
        if pair[1].startswith("JJ"):
            dic_of_pos["JJ"].append(pair[0])
        if pair[1].startswith("NN"):
            dic_of_pos["NN"].append(pair[0])
        if pair[1].startswith("RB"):
            dic_of_pos["RB"].append(pair[0])

    new_tokens_freq = nltk.FreqDist(new_tokens)
    print(new_tokens_freq.most_common(10))

    verbs = dic_of_pos["VB"]
    verb_freq = [(item, new_tokens_freq[item]) for item in verbs]
    sorted_verb_freq = sorted(verb_freq, key=lambda x: x[1], reverse=True)
    print(sorted_verb_freq[:10])

    adj = dic_of_pos["JJ"]
    adj_freq = [(item, freq[item]) for item in adj]
    sorted_adj_freq = sorted(adj_freq, key=lambda x: x[1], reverse=True)
    print(sorted_adj_freq[:30])

    nouns = dic_of_pos["NN"]
    noun_freq = [(item, freq[item]) for item in nouns]
    sorted_noun_freq = sorted(noun_freq, key=lambda x: x[1], reverse=True)
    print(sorted_noun_freq[:30])

    adverbs = dic_of_pos["RB"]
    adverb_freq = [(item, freq[item]) for item in adverbs]
    sorted_adverb_freq = sorted(adverb_freq, key=lambda x: x[1], reverse=True)
    print(sorted_adverb_freq[:10])

    # the argument asks for the portion of each list: given above
    def feature_giver(vec_of_which):
        ls = [sorted_noun_freq, sorted_verb_freq, sorted_adj_freq,
              sorted_adverb_freq, sorted_bigram_freq, sorted_trigram_freq]
        features = []
        for i in range(len(vec_of_which)):
            our_desired_ls = ls[i]
            features = features + our_desired_ls[:vec_of_which[i]]
        return list(set(features))

    important_features = feature_giver([100, 100, 130, 110, 100, 150])
    import_words = [item[0] for item in important_features]
    print(import_words[:10])

    def vec_to_str(inp):
        if isinstance(inp, tuple):
            res = ""
            for x in inp:
                for char in str(x):
                    res += char
                res += " "
        else:
            res = inp
        return res.strip()

    print(vec_to_str(('the', 'drive', 'through')))

    def feature_extractor(review):
        features = {}
        for item in import_words:
            item = vec_to_str(item)
            features["has(%s)" % item] = int(item in review)
        return features

    data = []  # the binary data and the label for each review
    data = data + [(feature_extractor(raw_text))]
    print("ahmed abohemeed====", data)
# end

    import pickle
    f = open("svm.pickle", "rb")
    # load : get the data from file
    model = pickle.load(f)
    f.close()

    prediction = model.classify_many(data)
    print("my prediction is ", prediction)
    # creates label and tells program where to load it
    firstLabel = tk.Label(window)
    firstLabel.config(text=str(prediction))  # alter the label to include text

    firstLabel.pack()  # loads the label to make it visible


# COMPARER FUNCTIONS
entry = Text(window, height=10)

button1 = Button(window, text="Classify review", command=classify_review,
                 width=12, bg='#03A9F4', fg='#fff')

# BUTTONS
entry.pack()
button1.pack()
window.mainloop()
