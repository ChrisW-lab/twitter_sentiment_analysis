'''Module containing all text preprocessing functions including vectorisation'''


import re
import os
import pandas as pd
import Levenshtein as lev

# processed_pickle_path = './annotated_tweets/consolidated_tweets.pkl'

# flags to expland what the '^' and '.' of the regular expression to include newlines
FLAGS = re.MULTILINE | re.DOTALL


def remove_duplicates(dataframe, threshold=0.5):
    '''Removes duplicate tweets from dataset using threshold ratio of Levenshtein distance to length of shortest tweet in comparison between two tweets
    :param: datframe; pandas dataframe with column 'text' (which includes all dataframe forms used in the project)
    :type: pandas dataframe
    :param: ratio; levenshtein distance divided by length of shortest tweets to get a 'percentage' difference between two tweets
    :return: pandas dataframe with duplicate tweets removed
    :rtype: pandas dataframe
    '''
    to_drop = []
    for index, row in dataframe.iterrows():
        for i, r in dataframe.iterrows():
            # retaining the first instance of the duplicate tweet
            if i > index:
                print(r)
                ratio = lev.distance(row['text'], r['text']) / min(len(row['text']), len(r['text']))
                if ratio < threshold:
                    to_drop.append(i)
    dataframe.drop(dataframe.index[to_drop], inplace=True)

    return dataframe


def bitcoin_token(text_string):
    '''Replaces various synonymous tokens with 'bitcoin'
    :param: text_string
    :type: string
    :return: altered_string
    :rtype: string
    '''
    bitcoin = re.compile(r'\S?bitcoin\S*', re.IGNORECASE)
    btc = re.compile(r'\S?btc\S*', re.IGNORECASE)
    altered_string = re.sub(bitcoin, 'btc', text_string)
    altered_string = re.sub(btc, 'btc', altered_string)
    return altered_string


def hashtag(match_obj):
    '''Tagging for hashtags dealing with capitalised and non-capitalised hashtags
    :param: match_obj
    :type: re match object
    :return: the appropriate processed and tagged hashtag
    :rtype: string
    '''

    # group() with no arguments returns the full matched string from the match object
    string = match_obj.group()
    # remove the '#'
    body = string[1:]
    # if all capitals tag the body
    if body.isupper():
        result = '<hashtag> {} <allcaps>'.format(body)
    else:
        result = " ".join(["<hashtag>"] + re.findall(r"[A-Z]?[a-z]*", body)).rstrip()

    return result


def allcaps(match_obj):
    '''Tags an re match object with <allcaps> to be used to tag words in tweets
    that are all capitals
    :param: match_obj
    :type: re match object
    :return: string with tag
    :rtype: string
    '''
    string = match_obj.group()
    return string.lower() + ' <allcaps>'


def tag(text_string):
    '''Function to apply standard text preprocessing to tweet as per GloVe preprocessing.  More detail in jupyter notebook "GloVe preprocessing explained"
    :param: text_string
    :type: string
    :return: list of strings
    :rtype: list (all elements string)
    '''
    # Different regex parts for smiley faces
    # the various forms of eyes in emoticons
    eyes = r"[8:=;]"
    # the various forms of noses in emoticons
    nose = r"['`\-]?"

    # helper function to shorten the code of re.sub(pattern, repl, text, flags)
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text_string, flags=FLAGS)

    # replace urls with tags
    text_string = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "<url>")
    # create whitespace around forward slashes
    text_string = re_sub(r"/", " / ")
    # replace twitter handles with tags
    text_string = re_sub(r"@\w+", "<user>")
    # replace the various emoticon faces with appropriate tags
    text_string = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>")
    text_string = re_sub(r"{}{}p+".format(eyes, nose), "<lolface>")
    text_string = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>")
    text_string = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>")
    text_string = re_sub(r"<3", "<heart>")
    # replacing numbers, hashtags, all capitals and repeated punctuation and letters with tags
    text_string = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>")
    # using the hashtag function defined
    text_string = re_sub(r"#\S+", hashtag)
    text_string = re_sub(r"([!?.]){2,}", r"\1 <repeat>")
    text_string = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>")
    text_string = re_sub(r"([A-Z]){2,}", allcaps)

    return text_string.lower()


def surplus_remove(text_string):
    '''Function to remove all non word characters apart from tag markers
    :param: text_string
    :type: string
    :return: clean_string
    :rtype: string
    '''
    # replace every character that is not one of those in the below regex
    punct = re.compile(r'[^a-zA-Z<>\s]+')
    # replace all whitespace characters with a simple space
    new_string = punct.sub('', text_string).rstrip()
    new_string = re.sub(r'\s+', ' ', new_string)
    return new_string


def preprocess(text_string):
    '''Combines functions to preprocess text strings according to reasoning outlined in report
    :param: text_string
    :type: string
    :return: clean_string
    :rtype: string
    '''
    clean_string = bitcoin_token(text_string)
    clean_string = tag(clean_string)
    clean_string = surplus_remove(clean_string)
    return clean_string


def tokenize(text_string):
    '''Preprocesses a text string and returns list of individual tokens (words)
    :param: text_string
    :type: string
    :return: tokens
    :rtype: list
    '''
    clean_string = preprocess(text_string)
    return clean_string.split(' ')


def process_csv(input_csv, output_pkl=None):
    '''Reads a csv of format output from annotation and writes a csv with both preprocessed and tokenized tweets
    :param: input_csv, filepath to csv with column headers: created_at, text, retweet_count, subjectivity, polarity
    :type: string
    :param: output_csv, filepath to csv which will have colun headers: created_at, text, preprocessed, tokens, subjectivity, polarity, retweet_count
    :type: string
    :return: void
    '''
    df = pd.read_csv(input_csv)
    # for row in df['text']:
    #     if type(row) != str:
    #         print(type(row))
    #         print(row)
    df['preprocessed'] = df['text'].apply(preprocess)
    df['tokens'] = df['text'].apply(tokenize)
    df = df[['created_at', 'text', 'preprocessed', 'tokens', 'subjectivity', 'polarity', 'retweet_count']]
    if output_pkl:
        df.to_pickle(output_pkl)
    return df


# if __name__ == '__main__':
#     df = process_csv('tweet_data/annotated_tweets/consolidated_tweets.csv', output_pkl='tweet_data/pickled_datasets/subjectivity_data.pkl')
#     print(df.shape)
