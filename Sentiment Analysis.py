import nltk
import random
import pickle
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers

    def classify(self,features):
        votes = []

        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self,features):
        votes = []

        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
        

short_pos = open("positive.txt","r").read()
short_neg = open("negative.txt","r").read()

documents =[]

for r in short_pos.split('\n'):
    documents.append((r,"pos"))

for r in short_neg.split('\n'):
    documents.append((r,"neg"))

all_words = []

short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())
    
all_words = nltk.FreqDist(all_words)
#print(all_words.most_common(15))

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev),category)
               for (rev,category) in documents]

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algorithm accuracy : ",(nltk.classify.accuracy(classifier,testing_set))*100)

classifier.show_most_informative_features(20)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB Algorithm accuracy : ",(nltk.classify.accuracy(MNB_classifier,testing_set))*100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB Algorithm accuracy : ",(nltk.classify.accuracy(BernoulliNB_classifier,testing_set))*100)
##LogisticRegression,SGDClassifier

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Algorithm accuracy : ",(nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100)

##SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
##SGDClassifier_classifier.train(training_set)
##print("SGDClassifier Algorithm accuracy : ",(nltk.classify.accuracy(SGDClassifier_classifier,testing_set))*100)




##SVC,LinearSVC,NuSVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC Algorithm accuracy : ",(nltk.classify.accuracy(SVC_classifier,testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Algorithm accuracy : ",(nltk.classify.accuracy(LinearSVC_classifier,testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Algorithm accuracy : ",(nltk.classify.accuracy(NuSVC_classifier,testing_set))*100)


voted_classifier = VoteClassifier(classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier,
                                  SVC_classifier,
                                  LogisticRegression_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier)

print("Voted Classifier accuracy : ",(nltk.classify.accuracy(voted_classifier,testing_set))*100)

print("Classification : ",voted_classifier.classify(testing_set[0][0]),"Confidence : ",voted_classifier.confidence(testing_set[0][0])*100)
print("Actual classification Result : ",testing[0][1])

print("Classification : ",voted_classifier.classify(testing_set[1][0]),"Confidence : ",voted_classifier.confidence(testing_set[1][0])*100)
print("Actual classification Result : ",testing[1][1])
print("Classification : ",voted_classifier.classify(testing_set[2][0]),"Confidence : ",voted_classifier.confidence(testing_set[2][0])*100)
print("Actual classification Result : ",testing[2][1])
print("Classification : ",voted_classifier.classify(testing_set[3][0]),"Confidence : ",voted_classifier.confidence(testing_set[3][0])*100)
print("Actual classification Result : ",testing[3][1])
print("Classification : ",voted_classifier.classify(testing_set[4][0]),"Confidence : ",voted_classifier.confidence(testing_set[4][0])*100)
print("Actual classification Result : ",testing[4][1])

