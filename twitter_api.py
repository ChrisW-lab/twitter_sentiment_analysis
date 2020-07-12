'''Module and script to stream Twitter data'''

import json
import csv
import ast
import time
import tweepy
import twitter_credentials as tc


#   assign consumer key and secret from twitter credentials stored separately
consumer_key, consumer_secret = tc.CONSUMER_KEY, tc.CONSUMER_SECRET_KEY
access_token, access_token_secret = tc.ACCESS_TOKEN, tc.ACCESS_TOKEN_SECRET


#create instances of Tweepy's authorisation class passing credentials
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# create API instance
api = tweepy.API(auth)


# create streaming class inheriting from Tweepy's StreamListener class
class MyStreamListener(tweepy.StreamListener):

    # initialise instance variable filename which will be the file where the streamed tweets are stored
    def __init__(self, filename):
        self.filename = filename
        super(MyStreamListener, self).__init__()

    # override on_status method which tells the StreamListener what to do when a filtered tweet is streamed
    def on_status(self, status):
        # identifies retweets and does not store them
        if hasattr(status, 'retweeted_status'):
            return
        status.text.encode('utf-8')
        if ('bitcoin' in status.text.lower()):
            with open(self.filename, 'a', encoding='utf-8') as stream:
                if hasattr(status, 'extended_tweet'):
                    row = [status.created_at, status.extended_tweet['full_text'], status.retweet_count]
                else:
                    row = [status.created_at, status.text, status.retweet_count]
                print(row)
                print(' ')
                csv_writer = csv.writer(stream)
                csv_writer.writerow(row)

    # override on_error method which tells StreamListener what to do when an error occurs
    def on_error(self, status_code):
        # this disconnects the stream if the status_code tells us its a rate limit (420)
        if status_code == 420:
            print('Error, rate limit hit.  Status code: ', status_code)
            return False


if __name__ == '__main__':
    # create instance of MyStreamListener class passing file path to store tweets
    myStreamListener = MyStreamListener('tweet_data/streamed_tweets_new.csv')
    # create instance of Stream passing authorised api, MyStreamListener instance and extended tweet parameter
    myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener, tweet_mode='extended')
    # filter the tweets using 'bitcoin' and 'btc' key terms
    myStream.filter(track=['bitcoin', 'btc'], languages=['en'])
