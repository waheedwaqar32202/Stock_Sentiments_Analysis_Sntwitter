# Import the libraries
import os
import sys
import warnings
import nltk
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(1500)

# Data Preprocessing and Feature Engineering
from textblob import TextBlob
import string


# newapi libraries
from newsapi.newsapi_client import NewsApiClient
from datetime import datetime
from datetime import date
from datetime import timedelta
# reddit library
# reddit libraries
import praw
# youtube api
from youtube_easy_api.easy_wrapper import *

import numpy as np
import matplotlib.pyplot as plt

# libraries for xgboost algorithm
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
#from xgboost import XGBRegressor

pd.options.mode.chained_assignment = None  # default='warn'

from twitter_sensor_memory import TwitterSensor
from news_sensor_memory import NewsSensor
from youtube_sensor_memory import YouTubeSensor
from reddit_sensor_memory import RedditSensor



def highlight_max(s):
    if s.dtype == np.object:
        is_neg = [False for _ in range(s.shape[0])]
    else:
        is_neg = s < 0
    return ['color: red;' if cell else 'color:black'
            for cell in is_neg]


# main method
if __name__ == '__main__':

    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_month_date = (datetime.now() - timedelta(30)).strftime('%Y-%m-%d %H:%M:%S')

    dt = date.today()
    today = datetime.combine(dt, datetime.min.time())
    yesterday = (today - timedelta(1)).strftime('%Y-%m-%d %H:%M:%S')
    last_two_days = (today - timedelta(2)).strftime('%Y-%m-%d %H:%M:%S')
    # new code here
    last_7_days = (today - timedelta(7)).strftime('%Y-%m-%d %H:%M:%S')
    last_14_days = (today - timedelta(14)).strftime('%Y-%m-%d %H:%M:%S')
    # end

    # read data from the excel file and store into stock_list"""

    ranking_df = pd.read_excel('stock_list.xlsx', engine='openpyxl')
    stock_list = ranking_df['Symbol']
    twitter = TwitterSensor()
    news = NewsSensor()
    youtube = YouTubeSensor()
    reddit = RedditSensor()


# get twitter data store into twitter_sentiments and save in cvs file
    twitter_sentiments = twitter.DoEveryThing(stock_list)
    reddit_sentiments = reddit.DoEveryThing(stock_list)
    news_sentiments = news.DoEveryThing(stock_list)
    youtube_sentiments = youtube.DoEveryThing(stock_list)

    if(twitter_sentiments.shape[0] != 0 and reddit_sentiments.shape[0] != 0 and news_sentiments.shape[0] != 0 and youtube_sentiments.shape[0] != 0):
        tw = twitter_sentiments[(twitter_sentiments['polarity'] != 0)]
        tw['publishedAt'] = pd.to_datetime(tw['publishedAt'], format="%Y-%m-%d %H:%M:%S")
        tw.sort_values(by='publishedAt')
        tw.to_csv('twitter_posts.csv', date_format="%Y-%m-%d %H:%M:%S")

        # get reddit data store into reddit_sentiments and save in cvs file

        rd = reddit_sentiments[(reddit_sentiments['polarity'] != 0)]
        rd.sort_values(by='publishedAt')
        rd.to_csv('reddit_posts.csv')

        # get  news data store into news_sentiments and save in cvs file

        ns = news_sentiments[(news_sentiments['polarity'] != 0)]
        ns.sort_values(by='publishedAt')
        ns.to_csv('news_posts.csv')

        # get youtube data store into youtube_sentiments and save in cvs file

        yt = youtube_sentiments[(youtube_sentiments['polarity'] != 0)]
        yt.sort_values(by='publishedAt')
        yt.to_csv('youtube_posts.csv')

        # read twitter,reddit,news and youtube data from all csv files
        twitter_data = pd.read_csv('twitter_posts.csv')
        news_data = pd.read_csv('news_posts.csv')
        reddit_data = pd.read_csv('reddit_posts.csv')
        youtube_data = pd.read_csv('youtube_posts.csv')

        # change column name tweet to title of twitter dataset
        twitter_data.rename(columns={'tweet': 'title'}, inplace=True)

        twitter_data['publishedAt'] = pd.to_datetime(twitter_data['publishedAt'])
        news_data['publishedAt'] = pd.to_datetime(news_data['publishedAt'])
        reddit_data['publishedAt'] = pd.to_datetime(reddit_data['publishedAt'])
        youtube_data['publishedAt'] = pd.to_datetime(youtube_data['publishedAt'])

        #last  14 days data

        tw_last_14_days = twitter_data[
            (twitter_data['publishedAt'] >= last_14_days) & (twitter_data['publishedAt'] <= today)]
        tw_last_14_days.to_csv("tw_last_14_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        ns_last_14_days = news_data[(news_data['publishedAt'] >= last_14_days) & (news_data['publishedAt'] <= today)]
        ns_last_14_days.to_csv("ns_last_14_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        rd_last_14_days = reddit_data[(reddit_data['publishedAt'] >= last_14_days) & (reddit_data['publishedAt'] <= today)]
        rd_last_14_days.to_csv("rd_last_14_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        yt_last_14_days = youtube_data[(youtube_data['publishedAt'] >= last_14_days) & (youtube_data['publishedAt'] <= today)]
        yt_last_14_days.to_csv("yt_last_14_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        # last 7 days data
        tw_last_7_days = twitter_data[(twitter_data['publishedAt'] >= last_7_days) & (twitter_data['publishedAt'] <= today)]
        tw_last_7_days.to_csv("tw_last_7_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        ns_last_7_days = news_data[(news_data['publishedAt'] >= last_7_days) & (news_data['publishedAt'] <= today)]
        ns_last_7_days.to_csv("ns_last_7_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        rd_last_7_days = reddit_data[ (reddit_data['publishedAt'] >= last_7_days) & (reddit_data['publishedAt'] <= today)]
        rd_last_7_days.to_csv("rd_last_7_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        yt_last_7_days = youtube_data[ (youtube_data['publishedAt'] >= last_7_days) & (youtube_data['publishedAt'] <= today)]
        yt_last_7_days.to_csv("yt_last_7_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        # three days data
        tw_last_two_days = twitter_data
        tw_last_two_days.to_csv("tw_last_two_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        ns_last_two_days = news_data[(news_data['publishedAt'] >= last_two_days) & (news_data['publishedAt'] <= today)]
        ns_last_two_days.to_csv("ns_last_two_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        rd_last_two_days = reddit_data[(reddit_data['publishedAt'] >= last_two_days) & (reddit_data['publishedAt'] <= today)]
        rd_last_two_days.to_csv("rd_last_two_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        yt_last_two_days = youtube_data[(youtube_data['publishedAt'] >= last_two_days) & (youtube_data['publishedAt'] <= today)]
        yt_last_two_days.to_csv("yt_last_two_days.csv", date_format="%Y-%m-%d %H:%M:%S")

        # yesterday

        tw_yesterday = twitter_data[(twitter_data['publishedAt'] >= yesterday) & (twitter_data['publishedAt'] <= today)]
        tw_yesterday.to_csv("tw_yesterday.csv", date_format="%Y-%m-%d %H:%M:%S")

        ns_yesterday = news_data[(news_data['publishedAt'] >= yesterday) & (news_data['publishedAt'] <= today)]
        ns_yesterday.to_csv("ns_yesterday.csv", date_format="%Y-%m-%d %H:%M:%S")

        rd_yesterday = reddit_data[(reddit_data['publishedAt'] >= yesterday) & (reddit_data['publishedAt'] <= today)]
        rd_yesterday.to_csv("rd_yesterday.csv", date_format="%Y-%m-%d %H:%M:%S")

        yt_yesterday = youtube_data[(youtube_data['publishedAt'] >= yesterday) & (youtube_data['publishedAt'] <= today)]
        yt_yesterday.to_csv("yt_yesterday.csv", date_format="%Y-%m-%d %H:%M:%S")
        # last 14 days Sentiments
        positive = [0] * len(stock_list)
        negative = [0] * len(stock_list)
        polarity = [0] * len(stock_list)
        total_post = [0] * len(stock_list)
        # create unique list of names
        UniqueNames_news = ns_last_14_days.keyword.unique()
        UniqueNames_reddit = rd_last_14_days.keyword.unique()
        UniqueNames_youtube = yt_last_14_days.keyword.unique()
        UniqueNames_twitter = tw_last_14_days.keyword.unique()

        # create a data frame dictionary to store your data frames
        DataFrameDict_news = {elem: pd.DataFrame for elem in UniqueNames_news}
        DataFrameDict_reddit = {elem: pd.DataFrame for elem in UniqueNames_reddit}
        DataFrameDict_youtube = {elem: pd.DataFrame for elem in UniqueNames_youtube}
        DataFrameDict_twitter = {elem: pd.DataFrame for elem in UniqueNames_twitter}

        for key in DataFrameDict_news.keys():
            DataFrameDict_news[key] = ns_last_14_days[:][ns_last_14_days.keyword == key]
        for key in DataFrameDict_reddit.keys():
            DataFrameDict_reddit[key] = rd_last_14_days[:][rd_last_14_days.keyword == key]
        for key in DataFrameDict_youtube.keys():
            DataFrameDict_youtube[key] = yt_last_14_days[:][yt_last_14_days.keyword == key]
        for key in DataFrameDict_twitter.keys():
            DataFrameDict_twitter[key] = tw_last_14_days[:][tw_last_14_days.keyword == key]

        # calculate positive, negative and neutral
        i = 0
        for stock in stock_list:
            stock = str(stock).lower()
            total_polarity = 0
            total_posts_count = 0
            if stock in UniqueNames_news:
                total_polarity += DataFrameDict_news[stock]['polarity'].sum()
                positive[i] += len(DataFrameDict_news[stock][DataFrameDict_news[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_news[stock])

            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_reddit:
                total_polarity += (DataFrameDict_reddit[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_reddit[stock][DataFrameDict_reddit[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_reddit[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_twitter:
                total_polarity += (DataFrameDict_twitter[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_twitter[stock][DataFrameDict_twitter[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_twitter[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_youtube:
                total_polarity += (DataFrameDict_youtube[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_youtube[stock][DataFrameDict_youtube[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_youtube[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            polarity[i] = total_polarity
            total_post[i] = total_posts_count

            if stock in UniqueNames_news:
                negative[i] += len(DataFrameDict_news[stock][DataFrameDict_news[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_reddit:
                negative[i] += len(DataFrameDict_reddit[stock][DataFrameDict_reddit[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_twitter:
                negative[i] += len(DataFrameDict_twitter[stock][DataFrameDict_twitter[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_youtube:
                negative[i] += len(DataFrameDict_youtube[stock][DataFrameDict_youtube[stock].polarity < 0])
            else:
                negative[i] += 0

            # craete each stock dataset by combining all data from all social sites and store in csv files
            if (stock in UniqueNames_youtube) and (stock in UniqueNames_twitter) and (stock in UniqueNames_reddit) and (
                    stock in UniqueNames_news):
                frames = [DataFrameDict_news[stock], DataFrameDict_reddit[stock], DataFrameDict_youtube[stock],
                          DataFrameDict_twitter[stock]]
                df_row_reindex = pd.concat(frames, join='inner', ignore_index=True)
                file_name = str(stock) + "_last_14_days.csv"
                df_row_reindex.to_csv(file_name)
            i = i + 1
        ranking_df['Stock Popularity'] = np.nan
        ranking_df['Last_14_Days_Ranking'] = np.nan
        ranking_df['Last_14_Days_Polarity'] = polarity
        ranking_df['Last_14_Days_Posts'] = total_post
        # Filled the Ranking column
        max_value = ranking_df['Last_14_Days_Posts'].max()

        for x in range(len(stock_list)):
            # Stock Popularity
            if ranking_df['Last_14_Days_Posts'][x] <= 20:
                ranking_df['Stock Popularity'][x] = 1
            else:
                ranking_df['Stock Popularity'][x] = 0

            # Stock Ranking
            if positive[x] > negative[x]:
                ranking_df['Last_14_Days_Ranking'][x] = "Positive"

            else:
                ranking_df['Last_14_Days_Ranking'][x] = "Negative"


        ranking_df['Last_14_Days_Positive'] = positive
        ranking_df['Last_14_Days_Negative'] = negative
        ranking_df['Last_14_Days_Score'] = ranking_df['Last_14_Days_Positive'] - ranking_df['Last_14_Days_Negative']

        """for x in range(len(stock_list)):
            if ranking_df['Stock Popularity'][x] == 0:
                ranking_df['Last_14_Days_Posts'][x] = "NULL"
                ranking_df['Last_14_Days_Ranking'][x] = "NULL"
                ranking_df['Last_14_Days_Positive'][x] = "NULL"
                ranking_df['Last_14_Days_Negative'][x] = "NULL"
                ranking_df['Last_14_Days_Score'][x] = "NULL"
                ranking_df['Last_14_Days_Polarity'][x] = "NULL"""





        s2 = ranking_df.style.apply(highlight_max)

        # Store stock ranking in csv file
        s2.to_excel("Stock_Ranking.xlsx")
        print("Last 14 days Done---> Your file is updated by filled Ranking column and store as Stock_Ranking.csv")



        # last 14 days Sentiments End

        # last 7 days Sentiments

        positive = [0] * len(stock_list)
        negative = [0] * len(stock_list)
        polarity = [0] * len(stock_list)
        total_post = [0] * len(stock_list)
        # create unique list of names
        UniqueNames_news = ns_last_7_days.keyword.unique()
        UniqueNames_reddit = rd_last_7_days.keyword.unique()
        UniqueNames_youtube = yt_last_7_days.keyword.unique()
        UniqueNames_twitter = tw_last_7_days.keyword.unique()

        # create a data frame dictionary to store your data frames
        DataFrameDict_news = {elem: pd.DataFrame for elem in UniqueNames_news}
        DataFrameDict_reddit = {elem: pd.DataFrame for elem in UniqueNames_reddit}
        DataFrameDict_youtube = {elem: pd.DataFrame for elem in UniqueNames_youtube}
        DataFrameDict_twitter = {elem: pd.DataFrame for elem in UniqueNames_twitter}

        for key in DataFrameDict_news.keys():
            DataFrameDict_news[key] = ns_last_7_days[:][ns_last_7_days.keyword == key]
        for key in DataFrameDict_reddit.keys():
            DataFrameDict_reddit[key] = rd_last_7_days[:][rd_last_7_days.keyword == key]
        for key in DataFrameDict_youtube.keys():
            DataFrameDict_youtube[key] = yt_last_7_days[:][yt_last_7_days.keyword == key]
        for key in DataFrameDict_twitter.keys():
            DataFrameDict_twitter[key] = tw_last_7_days[:][tw_last_7_days.keyword == key]

        # calculate positive, negative and neutral
        i = 0
        for stock in stock_list:
            stock = str(stock).lower()
            total_polarity = 0
            total_posts_count = 0
            if stock in UniqueNames_news:
                total_polarity += DataFrameDict_news[stock]['polarity'].sum()
                positive[i] += len(DataFrameDict_news[stock][DataFrameDict_news[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_news[stock])

            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_reddit:
                total_polarity += (DataFrameDict_reddit[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_reddit[stock][DataFrameDict_reddit[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_reddit[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_twitter:
                total_polarity += (DataFrameDict_twitter[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_twitter[stock][DataFrameDict_twitter[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_twitter[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_youtube:
                total_polarity += (DataFrameDict_youtube[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_youtube[stock][DataFrameDict_youtube[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_youtube[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            polarity[i] = total_polarity
            total_post[i] = total_posts_count

            if stock in UniqueNames_news:
                negative[i] += len(DataFrameDict_news[stock][DataFrameDict_news[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_reddit:
                negative[i] += len(DataFrameDict_reddit[stock][DataFrameDict_reddit[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_twitter:
                negative[i] += len(DataFrameDict_twitter[stock][DataFrameDict_twitter[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_youtube:
                negative[i] += len(DataFrameDict_youtube[stock][DataFrameDict_youtube[stock].polarity < 0])
            else:
                negative[i] += 0

            # craete each stock dataset by combining all data from all social sites and store in csv files
            if (stock in UniqueNames_youtube) and (stock in UniqueNames_twitter) and (stock in UniqueNames_reddit) and (
                    stock in UniqueNames_news):
                frames = [DataFrameDict_news[stock], DataFrameDict_reddit[stock], DataFrameDict_youtube[stock],
                          DataFrameDict_twitter[stock]]
                df_row_reindex = pd.concat(frames, join='inner', ignore_index=True)
                file_name = str(stock) + "_last_14_days.csv"
                df_row_reindex.to_csv(file_name)
            i = i + 1

        ranking_df['Last_7_Days_Ranking'] = np.nan
        ranking_df['Last_7_Days_Polarity'] = polarity
        ranking_df['Last_7_Days_Posts'] = total_post
        # Filled the Ranking column
        for x in range(len(stock_list)):
            if positive[x] > negative[x]:
                ranking_df['Last_7_Days_Ranking'][x] = "Positive"
            else:
                ranking_df['Last_7_Days_Ranking'][x] = "Negative"

        ranking_df['Last_7_Days_Positive'] = positive
        ranking_df['Last_7_Days_Negative'] = negative
        ranking_df['Last_7_Days_Score'] = ranking_df['Last_7_Days_Positive'] - ranking_df['Last_7_Days_Negative']
        s2 = ranking_df.style.apply(highlight_max)

        # Store stock ranking in csv file
        s2.to_excel("Stock_Ranking.xlsx")
        print("Last 7 days Done---> Your file is updated by filled Ranking column and store as Stock_Ranking.csv")

        # last 7 days Sentiments End






        # last three days Sentiments
        positive = [0] * len(stock_list)
        negative = [0] * len(stock_list)
        polarity = [0] * len(stock_list)
        total_post = [0] * len(stock_list)
        # create unique list of names
        UniqueNames_news = ns_last_two_days.keyword.unique()
        UniqueNames_reddit = rd_last_two_days.keyword.unique()
        UniqueNames_youtube = yt_last_two_days.keyword.unique()
        UniqueNames_twitter = tw_last_two_days.keyword.unique()

        # create a data frame dictionary to store your data frames
        DataFrameDict_news = {elem: pd.DataFrame for elem in UniqueNames_news}
        DataFrameDict_reddit = {elem: pd.DataFrame for elem in UniqueNames_reddit}
        DataFrameDict_youtube = {elem: pd.DataFrame for elem in UniqueNames_youtube}
        DataFrameDict_twitter = {elem: pd.DataFrame for elem in UniqueNames_twitter}

        for key in DataFrameDict_news.keys():
            DataFrameDict_news[key] = ns_last_two_days[:][ns_last_two_days.keyword == key]
        for key in DataFrameDict_reddit.keys():
            DataFrameDict_reddit[key] = rd_last_two_days[:][rd_last_two_days.keyword == key]
        for key in DataFrameDict_youtube.keys():
            DataFrameDict_youtube[key] = yt_last_two_days[:][yt_last_two_days.keyword == key]
        for key in DataFrameDict_twitter.keys():
            DataFrameDict_twitter[key] = tw_last_two_days[:][tw_last_two_days.keyword == key]

        # calculate positive, negative and neutral
        i = 0
        for stock in stock_list:
            stock = str(stock).lower()
            total_polarity = 0
            total_posts_count = 0
            if stock in UniqueNames_news:
                total_polarity += DataFrameDict_news[stock]['polarity'].sum()
                positive[i] += len(DataFrameDict_news[stock][DataFrameDict_news[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_news[stock])

            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_reddit:
                total_polarity += (DataFrameDict_reddit[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_reddit[stock][DataFrameDict_reddit[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_reddit[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_twitter:
                total_polarity += (DataFrameDict_twitter[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_twitter[stock][DataFrameDict_twitter[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_twitter[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_youtube:
                total_polarity += (DataFrameDict_youtube[stock]['polarity'] > 0).sum()
                positive[i] += len(DataFrameDict_youtube[stock][DataFrameDict_youtube[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_youtube[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            polarity[i] = total_polarity
            total_post[i] = total_posts_count

            if stock in UniqueNames_news:
                negative[i] += len(DataFrameDict_news[stock][DataFrameDict_news[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_reddit:
                negative[i] += len(DataFrameDict_reddit[stock][DataFrameDict_reddit[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_twitter:
                negative[i] += len(DataFrameDict_twitter[stock][DataFrameDict_twitter[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_youtube:
                negative[i] += len(DataFrameDict_youtube[stock][DataFrameDict_youtube[stock].polarity < 0])
            else:
                negative[i] += 0

            # craete each stock dataset by combining all data from all social sites and store in csv files
            if (stock in UniqueNames_youtube) and (stock in UniqueNames_twitter) and (stock in UniqueNames_reddit) and (
                    stock in UniqueNames_news):
                frames = [DataFrameDict_news[stock], DataFrameDict_reddit[stock], DataFrameDict_youtube[stock],
                          DataFrameDict_twitter[stock]]
                df_row_reindex = pd.concat(frames, join='inner', ignore_index=True)
                file_name = str(stock) + "_Weekly.csv"
                df_row_reindex.to_csv(file_name)
            i = i + 1

        ranking_df['Last_Two_Days_Ranking'] = np.nan
        ranking_df['Last_Two_Days_Polarity'] = polarity
        ranking_df['Last_Two_Days_Posts'] = total_post
        # Filled the Ranking column
        for x in range(len(stock_list)):
            if positive[x] > negative[x]:
                ranking_df['Last_Two_Days_Ranking'][x] = "Positive"
            else:
                ranking_df['Last_Two_Days_Ranking'][x] = "Negative"

        ranking_df['Last_Two_Days_Positive'] = positive
        ranking_df['Last_Two_Days_Negative'] = negative
        ranking_df['Last_Two_Days_Score'] = ranking_df['Last_Two_Days_Positive'] - ranking_df['Last_Two_Days_Negative']
        s2 = ranking_df.style.apply(highlight_max)

        # Store stock ranking in csv file
        s2.to_excel("Stock_Ranking.xlsx")
        print("Weekly Done---> Your file is updated by filled Ranking column and store as Stock_Ranking.csv")

        # Weekly End

        # Yesterday Sentiments
        positive = [0] * len(stock_list)
        negative = [0] * len(stock_list)
        polarity = [0] * len(stock_list)
        total_post = [0] * len(stock_list)
        # create unique list of names
        UniqueNames_news = ns_yesterday.keyword.unique()
        UniqueNames_reddit = rd_yesterday.keyword.unique()
        UniqueNames_youtube = yt_yesterday.keyword.unique()
        UniqueNames_twitter = tw_yesterday.keyword.unique()

        # create a data frame dictionary to store your data frames
        DataFrameDict_news = {elem: pd.DataFrame for elem in UniqueNames_news}
        DataFrameDict_reddit = {elem: pd.DataFrame for elem in UniqueNames_reddit}
        DataFrameDict_youtube = {elem: pd.DataFrame for elem in UniqueNames_youtube}
        DataFrameDict_twitter = {elem: pd.DataFrame for elem in UniqueNames_twitter}

        for key in DataFrameDict_news.keys():
            DataFrameDict_news[key] = ns_yesterday[:][ns_yesterday.keyword == key]
        for key in DataFrameDict_reddit.keys():
            DataFrameDict_reddit[key] = rd_yesterday[:][rd_yesterday.keyword == key]
        for key in DataFrameDict_youtube.keys():
            DataFrameDict_youtube[key] = yt_yesterday[:][yt_yesterday.keyword == key]
        for key in DataFrameDict_twitter.keys():
            DataFrameDict_twitter[key] = tw_yesterday[:][tw_yesterday.keyword == key]

        # calculate positive, negative and neutral
        i = 0
        for stock in stock_list:
            stock = str(stock).lower()
            total_polarity = 0
            total_posts_count = 0
            if stock in UniqueNames_news:
                total_polarity += DataFrameDict_news[stock]['polarity'].sum()
                positive[i] += len(DataFrameDict_news[stock][DataFrameDict_news[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_news[stock])

            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0
            if stock in UniqueNames_reddit:
                total_polarity += DataFrameDict_reddit[stock]['polarity'].sum()
                positive[i] += len(DataFrameDict_reddit[stock][DataFrameDict_reddit[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_reddit[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_twitter:
                total_polarity += DataFrameDict_twitter[stock]['polarity'].sum()
                positive[i] += len(DataFrameDict_twitter[stock][DataFrameDict_twitter[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_twitter[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            if stock in UniqueNames_youtube:
                total_polarity += DataFrameDict_youtube[stock]['polarity'].sum()
                positive[i] += len(DataFrameDict_youtube[stock][DataFrameDict_youtube[stock].polarity > 0])
                total_posts_count += len(DataFrameDict_youtube[stock])
            else:
                total_polarity += 0
                positive[i] += 0
                total_posts_count += 0

            polarity[i] = total_polarity
            total_post[i] = total_posts_count

            if stock in UniqueNames_news:
                negative[i] += len(DataFrameDict_news[stock][DataFrameDict_news[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_reddit:
                negative[i] += len(DataFrameDict_reddit[stock][DataFrameDict_reddit[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_twitter:
                negative[i] += len(DataFrameDict_twitter[stock][DataFrameDict_twitter[stock].polarity < 0])
            else:
                negative[i] += 0

            if stock in UniqueNames_youtube:
                negative[i] += len(DataFrameDict_youtube[stock][DataFrameDict_youtube[stock].polarity < 0])
            else:
                negative[i] += 0

            i = i + 1

        ranking_df['Yesterday_Ranking'] = np.nan
        ranking_df['Yesterday_Polarity'] = polarity
        ranking_df['Yesterday_Total_Posts'] = total_post
        # Filled the Ranking column
        for x in range(len(stock_list)):
            if positive[x] > negative[x]:
                ranking_df['Yesterday_Ranking'][x] = "Positive"
            else:
                ranking_df['Yesterday_Ranking'][x] = "Negative"

        ranking_df['Yesterday_Positive'] = positive
        ranking_df['Yesterday_Negative'] = negative
        ranking_df['Yesterday_Score'] = ranking_df['Yesterday_Positive'] - ranking_df['Yesterday_Negative']

        # delete Ranking column from ranking-df
        ranking_df = ranking_df.drop('Ranking', 1)
        ranking_df.sort_values(by='Yesterday_Polarity')
        s2 = ranking_df.style.apply(highlight_max)
        # Store stock ranking in csv file
        s2.to_excel("Stock_Ranking.xlsx")
        print("Yesterday Done---> Your file is updated by filled Ranking column and store as Stock_Ranking.csv")

        #cell format
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter('Stock_Ranking.xlsx', engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object.
        df = s2
        df.to_excel(writer, sheet_name='Sheet1')

        # Get the xlsxwriter workbook and worksheet objects.
        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        # Add a header format.
        header_format = workbook.add_format({
            'bold': True,
            'fg_color': '#b4cc81',
            'border': 1})
        header_format1 = workbook.add_format({
            'bold': True,
            'fg_color': '#eee8aa',
            'border': 1})
        header_format2 = workbook.add_format({
            'bold': True,
            'fg_color': '#d0f0c0',
            'border': 1})

        # Write the column headers with the defined format.
        count = 0
        for col_num, value in enumerate(df.columns.values):
            if count < 2:
                count = count+1
                worksheet.write(0, col_num + 1, value, header_format1)
            elif count > 7:
                worksheet.write(0, col_num + 1, value, header_format2)
            else:
                worksheet.write(0, col_num + 1, value, header_format)
                count = count+1

        # Close the Pandas Excel writer and output the Excel file.
        writer.save()


    # Yesterday End
    else:
        if twitter_sentiments.shape[0] != 0:
            print("Sorry--->  Twitter data not found against stocks")
        if reddit_sentiments.shape[0] != 0:
            print("Sorry--->  Reddit data not found against stocks")
        if news_sentiments.shape[0] != 0:
            print("Sorry--->  News data not found against stocks")
        if youtube_sentiments.shape[0] != 0:
            print("Sorry--->  Youtube data not found against stocks")


