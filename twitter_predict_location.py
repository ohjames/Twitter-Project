import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

new_york = pd.read_json('new_york.json', lines = True)
london = pd.read_json('london.json', lines = True)
paris = pd.read_json('paris.json', lines = True)
print(len(new_york))
print(new_york.columns)
print(len(london))
print(london.columns)
print(len(paris))
print(paris.columns)
# combine df together
new_york_text = new_york.text.tolist()
london_text = london.text.tolist()
paris_text = paris.text.tolist()
all_texts = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)
# train/test data
train_data, test_data, train_label, test_label = train_test_split(all_texts, labels, test_size=0.2, random_state=1)
print(len(train_data))
print(len(test_data))
# naive bayes classifier method
# count vector
counter = CountVectorizer()
counter.fit(train_data)
train_count = counter.transform(train_data)
test_count = counter.transform(test_data)
# classify
classifier = MultinomialNB()
classifier.fit(train_count, train_label)
predictions = classifier.predict(test_count)
print(accuracy_score(test_label, predictions))
# confusion matrix
print(confusion_matrix(test_label, predictions))
# test tweet
tweet = 'I love Big Ben'
tweet_counts = counter.transform([tweet])
print(classifier.predict(tweet_counts))
#todo try to make better train/test sets