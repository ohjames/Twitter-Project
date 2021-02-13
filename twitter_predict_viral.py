import pandas as pd
import numpy as np
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

tweets = pd.read_json('random_tweets.json', lines=True)
print(len(tweets))
print(tweets.columns)
print(tweets.head())

# predict whether a tweet will go viral
# choose median because mean does not capture outliers
tweets['is_viral'] = np.where(tweets.retweet_count >= np.median(tweets.retweet_count), 1, 0)
print(tweets.is_viral.value_counts())
# choose features we think will affect going viral
tweets['tweet_length'] = tweets.apply(lambda x: len(x['text']), axis = 1)
tweets['follower_count'] = tweets.apply(lambda x: x['user']['followers_count'], axis = 1)
tweets['friend_count'] = tweets.apply(lambda x: x['user']['friends_count'], axis = 1)
# normalize data
labels = tweets.is_viral
data = tweets[['tweet_length', 'follower_count', 'friend_count']]
scale_data = scale(data, axis = 0)
print(scale_data[0])
# train/test set
X_train, X_test, y_train, y_test = train_test_split(scale_data, labels, test_size = 0.2, random_state = 1)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
# try different k values
scores = []
for k in range(1, 200):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    scores.append(classifier.score(X_test, y_test))
plt.plot(range(1,200), scores)
plt.xlabel('k values')
plt.ylabel('scores')
plt.show()
#todo improve model and add more robust features