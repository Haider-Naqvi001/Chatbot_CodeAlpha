import nltk
import os
import csv
from nltk.stem.snowball import SnowballStemmer
import random
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import warnings

warnings.filterwarnings("ignore")
nltk.download("popular")
nltk.download('averaged_perceptron_tagger')

lmtzr = WordNetLemmatizer()
stemmer = SnowballStemmer("english")

def preprocess(sentence):
    sentence = sentence.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_words = [w for w in tokens if w not in stopwords.words("english")]
    return filtered_words

def extract_tagged(tags):
    features = []
    for tagged_word in tags:
        word, tag = tagged_word
        if tag in ["NN", "VBN", "NNS", "VBP", "RB", "VBZ", "VBG", "PRP", "JJ"]:
            features.append(word)
    return features

def extract_feature(text):
    words = preprocess(text)
    tags = nltk.pos_tag(words)
    extracted_feature = extract_tagged(tags)
    stemmed_words = [stemmer.stem(x) for x in extracted_feature]
    result = [lmtzr.lemmatize(x) for x in stemmed_words]
    return result

def word_feats(words):
    return dict([(word, True) for word in words])


def get_content(filename):
    doc = os.path.abspath(filename)
    if not os.path.exists(doc):
        raise FileNotFoundError(f"File not found: {doc}")
    with open(doc, "r", encoding="utf-8") as content_file:
        lines = csv.reader(content_file, delimiter="|")
        data = [x for x in lines if len(x) == 3]
        return data

def split_data(data, split_ratio):
    random.shuffle(data)
    data_length = len(data)
    train_split = int(data_length * split_ratio)
    return data[:train_split], data[train_split:]


filename = "Script.txt"
data = get_content(filename)


def extract_feature_from_doc(data):
    result = []
    corpus = []
    answers = {}
    for (text, category, answer) in data:
        features = extract_feature(text)
        corpus.append(features)
        result.append((word_feats(features), category))
        answers[category] = answer
    return result, sum(corpus, []), answers

features_data, corpus, answers = extract_feature_from_doc(data)
training_data, test_data = split_data(features_data, 0.8)

def train_using_decision_tree(training_data, test_data):
    classifier = nltk.classify.DecisionTreeClassifier.train(training_data, entropy_cutoff=0.6, support_cutoff=6)
    training_set_accuracy = nltk.classify.accuracy(classifier, training_data)
    print("Training set accuracy: ", training_set_accuracy)
    test_set_accuracy = nltk.classify.accuracy(classifier, test_data)
    print("Test set accuracy: ", test_set_accuracy)
    classifier_name = "Decision Tree Classifier"
    return classifier, classifier_name, test_set_accuracy, training_set_accuracy

dtclassifier, classifier_name, test_set_accuracy, training_set_accuracy = train_using_decision_tree(training_data, test_data)

def get_response(classifier, sentence):
    features = word_feats(extract_feature(sentence))
    prediction = classifier.classify(features)
    if prediction:
        return prediction
    else:
        return "Sorry, I didn't understand that. Can you rephrase?"

def reply(input_sentence):
    category = dtclassifier.classify(word_feats(extract_feature(input_sentence)))
    return answers.get(category, "Sorry, I don't have an answer for that.")

while True:
    input_sentence = input("Enter a sentence (or type 'exit' to quit): ")
    if input_sentence.lower() == 'exit':
        print("Goodbye!")
        break
    response = reply(input_sentence)
    print(response)
