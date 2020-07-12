'''Tests for all preprocessing functions'''
import utils_preprocessing as upp
import pandas as pd


def test_remove_duplicates1():
    testdf = pd.read_csv('unit_testing_data/streamed_tweets_test.csv')
    testdf_dr = pd.read_csv('unit_testing_data/streamed_tweets_test_dr.csv')
    removed = upp.remove_duplicates(testdf)
    assert(list(testdf_dr['text']) == list(removed['text']))


def test_remove_duplicates2():
    testdf = pd.read_csv('unit_testing_data/short_csv_test.csv')
    testdf_dr = pd.read_csv('unit_testing_data/short_csv_test_dr.csv')
    removed = upp.remove_duplicates(testdf)
    assert(list(testdf_dr['text']) == list(removed['text']))


def test_bitcoin_token1():
    text_string = '@bitcoin @BITCOIN #bitcoin #BITCOIN'
    assert(upp.bitcoin_token(text_string) == 'btc btc btc btc')


def test_bitcoin_token2():
    text_string = 'BTC/USD, btc/usd'
    assert(upp.bitcoin_token(text_string) == 'btc btc')


def test_bitcoin_token3():
    text_string = 'BTC/USD is getting stronger'
    assert(upp.bitcoin_token(text_string) == 'btc is getting stronger')


def test_tag1():
    text_string = ':-)'
    assert(upp.tag(text_string) == '<smile>')


def test_tag2():
    text_string = 'wayyyy :-p'
    assert(upp.tag(text_string) == 'way <elong> <lolface>')


def test_tag3():
    text_string = 'I TEST alllll kinds of #hashtags and #HASHTAGS, @mentions and 3000 (http://t.co/dkfjkdf). w/ <3 :) haha!!!!!'
    assert(upp.tag(text_string) == 'i test <allcaps> al <elong> kinds of <hashtag> hashtags and <hashtag> hashtags <allcaps>, <allcaps> <user> and <number> (<url>). w /  <heart> <smile> haha! <repeat>')


def test_tag_emoji():
    '''ensures emojis are not getting tagged'''
    emoji_string = 'btc 😂'
    assert(upp.tag(emoji_string) == 'btc 😂')


def test_surplus_remove1():
    text_string = 'can\'t'
    assert(upp.surplus_remove(text_string) == 'cant')


def test_surplus_remove2():
    text_string = 'i\'m checking punc_remove!'
    assert(upp.surplus_remove(text_string) == 'im checking puncremove')


def test_surplus_remove3():
    emoji_string = 'BTC 😂'
    assert(upp.surplus_remove(emoji_string) == 'BTC')


def test_surplus_remove4():
    text_string = 'i test <allcaps> al <elong> kinds of <hashtag> hashtags and <hashtag> hashtags <allcaps>, <allcaps> <user> and <number> (<url>). w /  <heart> <smile> haha! <repeat>'
    assert(upp.surplus_remove(text_string) == 'i test <allcaps> al <elong> kinds of <hashtag> hashtags and <hashtag> hashtags <allcaps> <allcaps> <user> and <number> <url> w <heart> <smile> haha <repeat>')


def test_pre_process1():
    text_string = 'BTC/USD going UP'
    assert(upp.preprocess(text_string) == 'btc going up <allcaps>')


def test_pre_process2():
    text_string = 'WOW look at #Bitcoin, the price is rocketting! #BullMarket'
    assert(upp.preprocess(text_string) == 'wow <allcaps> look at btc the price is rocketting <hashtag> bull market')


def test_pre_process():
    text_string = 'btc/usd taking a hammering today. More bad news coming with Fed pushing regulation'
    assert(upp.preprocess(text_string) == 'btc taking a hammering today more bad news coming with fed pushing regulation')


def test_preprocess_edge1():
    text_string = '248264923, 32dfsfidub qdjasibd btc adaisjdb bitcoin'
    assert(upp.preprocess(text_string) == '<number> <number>dfsfidub qdjasibd btc adaisjdb btc')


def test_preprocess_edge2():
    text_string = '       '
    assert(upp.preprocess(text_string) == '')


def test_preprocess_junk():
    text_string = '#buybuybuy #products $$$cheap'
    assert(upp.preprocess(text_string) == '<hashtag> buybuybuy <hashtag> products cheap')
