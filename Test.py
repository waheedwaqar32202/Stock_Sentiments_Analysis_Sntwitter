# Import the libraries
import sys
import warnings
import nltk
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)
sys.setrecursionlimit(1500)

# Data Preprocessing and Feature Engineering
from textblob import TextBlob
import string

# twint libraries
import twint

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
from xgboost import XGBRegressor

pd.options.mode.chained_assignment = None  # default='warn'

from twitter_sensor_memory import TwitterSensor
from news_sensor_memory import NewsSensor
from youtube_sensor_memory import YouTubeSensor
from reddit_sensor_memory import RedditSensor


# series to supervised data
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# stock prediction function
def stock_prediction(stock_data):
    # transform a time series dataset into a supervised learning dataset

    # load the dataset
    series = stock_data
    values = series.values
    # transform the time series data into supervised learning
    train = series_to_supervised(values, n_in=6)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(trainX, trainy)
    # construct an input for a new preduction
    row = values[-6:].flatten()
    # make a one-step prediction
    yhat = model.predict(asarray([row]))

    return yhat


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
    week = (today - timedelta(2)).strftime('%Y-%m-%d %H:%M:%S')

    # read data from the excel file and store into stock_list"""

    ranking_df = pd.read_excel('stock_list.xlsx', engine='openpyxl')
    stock_list = ranking_df['Symbol']
    twitter = TwitterSensor()
    news = NewsSensor()
    youtube = YouTubeSensor()
    reddit = RedditSensor()


# get twitter data store into twitter_sentiments and save in cvs file
    twitter_sentiments = twitter.DoEveryThing(stock_list)
    tw = twitter_sentiments[(twitter_sentiments['polarity'] != 0)]
    tw.sort_values(by='publishedAt')
    tw.to_csv('twitter_posts.csv', date_format="%Y-%m-%d %H:%M:%S")

    # get reddit data store into reddit_sentiments and save in cvs file
    reddit_sentiments = reddit.DoEveryThing(stock_list)
    rd = reddit_sentiments[(reddit_sentiments['polarity'] != 0)]
    rd.sort_values(by='publishedAt')
    rd.to_csv('reddit_posts.csv')

    # get  news data store into news_sentiments and save in cvs file
    news_sentiments = news.DoEveryThing(stock_list)
    ns = news_sentiments[(news_sentiments['polarity'] != 0)]
    ns.sort_values(by='publishedAt')
    ns.to_csv('news_posts.csv')

    # get youtube data store into youtube_sentiments and save in cvs file
    youtube_sentiments = youtube.DoEveryThing(stock_list)
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


    # three days data
    tw_weekly = twitter_data
    tw_weekly.to_csv("tw_weekly.csv", date_format="%Y-%m-%d %H:%M:%S")

    ns_weekly = news_data[(news_data['publishedAt'] >= week) & (news_data['publishedAt'] <= today)]
    ns_weekly.to_csv("ns_weekly.csv", date_format="%Y-%m-%d %H:%M:%S")

    rd_weekly = reddit_data[(reddit_data['publishedAt'] >= week) & (reddit_data['publishedAt'] <= today)]
    rd_weekly.to_csv("rd_weekly.csv", date_format="%Y-%m-%d %H:%M:%S")

    yt_weekly = youtube_data[(youtube_data['publishedAt'] >= week) & (youtube_data['publishedAt'] <= today)]
    yt_weekly.to_csv("yt_weekly.csv", date_format="%Y-%m-%d %H:%M:%S")

    # yesterday

    tw_yesterday = twitter_data[(twitter_data['publishedAt'] >= yesterday) & (twitter_data['publishedAt'] <= today)]
    tw_yesterday.to_csv("tw_yesterday.csv", date_format="%Y-%m-%d %H:%M:%S")

    ns_yesterday = news_data[(news_data['publishedAt'] >= yesterday) & (news_data['publishedAt'] <= today)]
    ns_yesterday.to_csv("ns_yesterday.csv", date_format="%Y-%m-%d %H:%M:%S")

    rd_yesterday = reddit_data[(reddit_data['publishedAt'] >= yesterday) & (reddit_data['publishedAt'] <= today)]
    rd_yesterday.to_csv("rd_yesterday.csv", date_format="%Y-%m-%d %H:%M:%S")

    yt_yesterday = youtube_data[(youtube_data['publishedAt'] >= yesterday) & (youtube_data['publishedAt'] <= today)]
    yt_yesterday.to_csv("yt_yesterday.csv", date_format="%Y-%m-%d %H:%M:%S")


    # Weekly Sentiments
    positive = [0] * len(stock_list)
    negative = [0] * len(stock_list)
    polarity = [0] * len(stock_list)
    total_post = [0] * len(stock_list)
    # create unique list of names
    UniqueNames_news = ns_weekly.keyword.unique()
    UniqueNames_reddit = rd_weekly.keyword.unique()
    UniqueNames_youtube = yt_weekly.keyword.unique()
    UniqueNames_twitter = tw_weekly.keyword.unique()



    # create a data frame dictionary to store your data frames
    DataFrameDict_news = {elem: pd.DataFrame for elem in UniqueNames_news}
    DataFrameDict_reddit = {elem: pd.DataFrame for elem in UniqueNames_reddit}
    DataFrameDict_youtube = {elem: pd.DataFrame for elem in UniqueNames_youtube}
    DataFrameDict_twitter = {elem: pd.DataFrame for elem in UniqueNames_twitter}

    for key in DataFrameDict_news.keys():
        DataFrameDict_news[key] = ns_weekly[:][ns_weekly.keyword == key]
    for key in DataFrameDict_reddit.keys():
        DataFrameDict_reddit[key] = rd_weekly[:][rd_weekly.keyword == key]
    for key in DataFrameDict_youtube.keys():
        DataFrameDict_youtube[key] = yt_weekly[:][yt_weekly.keyword == key]
    for key in DataFrameDict_twitter.keys():
        DataFrameDict_twitter[key] = tw_weekly[:][tw_weekly.keyword == key]

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
        if (stock in UniqueNames_youtube) and (stock in UniqueNames_twitter) and (stock in UniqueNames_reddit) and (stock in UniqueNames_news):
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

# Yesterday End

#Stock Prediction
    pred_value = []
    for stock in stock_list:
        file_name = str(stock) + "_monthly.csv"
        data = pd.read_csv(file_name)
        data.drop(data.columns.difference(['polarity']), 1, inplace=True)
        data = data[data.polarity != 0]
        value = stock_prediction(data)
        print(value)
        if value > 0:
            pred_value.append('Positive')
        elif value < 0:
            pred_value.append('Negative')
        else:
            pred_value.append('Neutral')

    # prediction  dataframe and store into csv file.
    prediction = pd.DataFrame(list(zip(stock_list, pred_value)), columns=['Stock', 'Next Day Prediction'])
    prediction.to_csv('Prediction.csv')
    print("Done--->  Prediction is saved into Prediction.csv file ")
    print("Sentiments Analysis Completed")