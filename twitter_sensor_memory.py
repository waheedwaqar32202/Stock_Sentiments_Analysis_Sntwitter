import sys
import warnings
import nltk
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(1500)
from datetime import datetime

from datetime import timedelta
# Data Preprocessing and Feature Engineering
from textblob import TextBlob
import string
from datetime import datetime

import snscrape.modules.twitter as sntwitter
class TwitterSensor:
    def FetchSensorData(self, keyword):
        today_date = datetime.now()
        today_date = datetime.strftime(today_date, '%Y-%m-%d')
        start_date = datetime.now() - timedelta(13)
        start_date = datetime.strftime(start_date, '%Y-%m-%d')
        tweets_list2 = []
        # Using TwitterSearchScraper to scrape data and append tweets to list
        key = keyword + " stock"
        print(key)
        for i, tweet in enumerate(
                sntwitter.TwitterSearchScraper(key+' near:"US" since:'+start_date+' until:'+today_date).get_items()):

            tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

        # Creating a dataframe from the tweets list above
        print("stock " + keyword + " completed ----> wait please")
        tweets_df = pd.DataFrame(tweets_list2, columns=['publishedAt', 'tweet id', 'tweet', 'username'])


        tweets_df['publishedAt'] = pd.to_datetime(tweets_df['publishedAt'])
        tweets_df["keyword"] = str(keyword).lower()
        return tweets_df


    def dataCleaning(self, text):
            from nltk.corpus import stopwords
            punctuation = string.punctuation
            stopwords = stopwords.words('english')
            text = text.lower()
            text = "".join(x for x in text if x not in punctuation)
            words = text.split()
            words = [w for w in words if w not in stopwords]
            text = " ".join(words)

            return text


    def DoLowLevelPerception(self, posts: pd.DataFrame) -> pd.DataFrame:
        if posts.shape[0] == 0:
            return posts
        # Clean title and make another coulmn to store cleaned title
        posts['cleaned_tweet'] = posts['tweet'].apply(self.dataCleaning)
        # calculate polarity and subjectivity of title using textblob
        posts['polarity'] = posts['cleaned_tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
        posts['subjectivity'] = posts['cleaned_tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.subjectivity)
        return posts


    def DoEveryThing(self, stock_list):

        posts = []
        for stock in stock_list:
            df = self.FetchSensorData(stock)
            post = self.DoLowLevelPerception(df)
            posts.append(post)

        posts = pd.concat(posts, ignore_index=True)

        return posts
# main method
if __name__ == '__main__':
    pass

