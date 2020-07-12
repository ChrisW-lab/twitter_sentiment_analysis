'''Unit tests for utils_vectorisation functions'''

import utils_vectorisation as uv
import numpy as np
import pandas as pd


def test_normalise1():
    ary = np.array([[1, 1, 1], [1, 1, 1]])
    assert(uv.normalise(ary).all() == np.array([[1, 1, 1], [1, 1, 1]]).all())


def test_normlise2():
    ary = np.array([[2, 2, 2], [4, 4, 4], [4, 4, 4]])
    assert(uv.normalise(ary).all() == np.array([[0.5, 0.5, 0.5], [1, 1, 1], [1, 1, 1]]).all())


def test_normalise3():
    ary = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert(uv.normalise(ary).all() == np.array([[0.33333333, 0.33333333, 0.33333333], [0.66666667, 0.66666667, 0.66666667], [1.0, 1.0, 1.0]]).all())


def test_create_raw_X_Y1():
    df = pd.DataFrame({'subjectivity': [1, 0], 'polarity': [1, None], 'tokens': ['hello', 'world']})
    assert(uv.create_raw_X_Y(df, classification='polarity') == (['hello'], [1]))


def test_create_raw_X_Y2():
    df = pd.DataFrame({
        'subjectivity': [1, 1, 1, 0],
        'polarity': [1, 0, 1, None],
        'tokens': ['<hastag>', '<allcaps>', 'bitcoin', 'nonsense']
    })
    assert(uv.create_raw_X_Y(df, classification='polarity') == (['<hastag>', '<allcaps>', 'bitcoin'], [1, 0, 1]))


def test_create_raw_X_Y3():
    df = pd.DataFrame({
        'subjectivity': [1, 1, 1, 0],
        'polarity': [1, 0, 1, None],
        'tokens': ['<hastag>', '<allcaps>', 'bitcoin', 'nonsense']
    })
    assert(uv.create_raw_X_Y(df, classification='subjectivity') == (['<hastag>', '<allcaps>', 'bitcoin', 'nonsense'], [1, 1, 1, 0]))


def test_create_raw_X_Y4():
    df = pd.DataFrame({
        'tokens': [['a', 'b'], ['c', 'd']],
        'subjectivity': [1, 0]
    })
    assert(uv.create_raw_X_Y(df)[0] == [['a', 'b'], ['c', 'd']])


def test_split_train_test1():
    X = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    Y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    for i in range(100):
        (X_train, Y_train), (X_test, Y_test) = uv.split_train_test(X, Y)
        assert(len(X_train) == 9)
        assert(len(X_test) == 1)
        assert(len(Y_train) == 9)
        assert(len(Y_test) == 1)


def test_balance_data1():
    X = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    Y = np.array([1, 0, 1])
    X_balanced, Y_balanced = uv.balance_data(X, Y)
    unique, counts = np.unique(Y_balanced, return_counts=True)
    assert(counts.all() == np.array([1, 1]).all())


def test_balance_data2():
    X = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    Y = np.array([1, 0, 1, 1])
    X_balanced, Y_balanced = uv.balance_data(X, Y)
    unique, counts = np.unique(Y_balanced, return_counts=True)
    assert(counts.all() == np.array([1, 1]).all())


def test_balance_data3():
    X = np.array([[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
    Y = np.array([1, 0, 1, 1])
    X_balanced, Y_balanced = uv.balance_data(X, Y)
    index = np.where(Y_balanced == 0)
    ref = np.array([0, 0, 0, 0])
    assert(X_balanced[index].all() == ref.all())


def test_balance_data4():
    X = np.random.rand(20, 5)
    Y = np.concatenate([np.zeros(2), np.ones(18)])
    X_balanced, Y_balanced = uv.balance_data(X, Y)
    assert(np.size(Y_balanced) == 4)


def test_balance_data5():
    X = np.random.rand(20, 5)
    Y = np.concatenate([np.zeros(2), np.ones(18)])
    X_balanced, Y_balanced = uv.balance_data(X, Y)
    unique, counts = np.unique(Y_balanced, return_counts=True)
    d = dict(zip(unique, counts))
    assert(d[0] == 2)


def test_balance_data6():
    X = np.random.rand(20, 5)
    Y = np.concatenate([np.zeros(2), np.ones(18)])
    X_balanced, Y_balanced = uv.balance_data(X, Y)
    unique, counts = np.unique(Y_balanced, return_counts=True)
    d = dict(zip(unique, counts))
    assert(d[1] == 2)


def test_balance_data_types1():
    raw_X = [['string1', 'string2'], ['string3', 'string4']]
    raw_Y = [1, 0]
    X, Y = uv.balance_data(raw_X, raw_Y)
    assert(type(X[0][0]) is np.str_)


def test_token_index_map1():
    X = np.array([['string1', 'string2'], ['string3']])
    timap = uv.token_index_map(X)
    assert(timap == {'string1': 0, 'string2': 1, 'string3': 2})


def test_bow1():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello'], ['world']]
    })
    expected_vectors = np.array([[0], [1]])
    expected_Y = np.array(df['subjectivity'])
    bow_vectors, Y = uv.bow(df, target='subjectivity')
    assert(bow_vectors.all() == expected_vectors.all())
    assert(Y.all() == expected_Y.all())


def test_bow2():
    df = pd.DataFrame({
        'tokens': [['<allcaps>', 'bitcoin', 'btc'], ['btc', 'bitcoin'], ['<hashtag>', 'btc']],
        'subjectivity': [1, 1, 0],
        'polarity': [1, 0, None]
    })
    vectors, Y = uv.bow(df, target='subjectivity')
    assert(vectors.shape[0] == 2)


def test_bow2():
    df = pd.DataFrame({
        'tokens': [['<allcaps>', 'bitcoin', '<url>'], ['<url>', 'bitcoin'], ['<hashtag>', '<url>']],
        'subjectivity': [1, 1, 0],
        'polarity': [1, 0, None]
    })
    vectors, Y = uv.bow(df, target='polarity')
    assert(vectors.shape == (2, 3))


def test_bow3():
    df = pd.DataFrame({
        'tokens': [['<allcaps>', 'bitcoin', '<url>'], ['<url>', 'bitcoin'], ['<hashtag>', '<url>']],
        'subjectivity': [1, 1, 0],
        'polarity': [1, 0, None]
    })
    for i in range(100):
        vectors, Y = uv.bow(df, target='polarity')
        assert(vectors.all() <= 1)


def test_bow4():
    df = pd.DataFrame({
        'tokens': [['<allcaps>', 'bitcoin', '<url>'], ['<url>', 'bitcoin'], ['<hashtag>', '<url>']],
        'subjectivity': [1, 1, 0],
        'polarity': [1, 0, None]
    })
    for i in range(100):
        vectors, Y = uv.bow(df, target='subjectivity')
        index = np.where(Y == 0)[0][0]
        assert(sum(vectors[index]) == 2)


def test_sequence_bow1():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello', 'hello'], ['world', 'world']]
    })
    expected_seq_shape = (2, 2, 2)
    for i in range(100):
        sequence_bow, Y = uv.sequence_bow(2, df, target='subjectivity')
        assert(sequence_bow.shape == expected_seq_shape)


def test_sequence_bow2():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello', 'hello'], ['world', 'world']]
    })
    expected_seq_shape = (2, 3, 2)
    for i in range(100):
        sequence_bow, Y = uv.sequence_bow(3, df, target='subjectivity')
        assert(sequence_bow.shape == expected_seq_shape)


def test_sequence_bow3():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello', 'hello'], ['world', 'world']]
    })
    for i in range(100):
        sequence_bow, Y = uv.sequence_bow(3, df, target='subjectivity')
        assert(sequence_bow[np.random.randint(2)][2].all() == np.zeros(2).all())


def test_sequence_bow4():
    df = pd.DataFrame({
        'subjectivity': [1, 0, 1, 0],
        'polarity': [1, None, 1, 1],
        'tokens': [['hello', 'hello'], ['world', 'world', 'party', 'time'], ['hello', 'party'], ['hello']]
    })
    for i in range(100):
        sequence_bow, Y = uv.sequence_bow(3, df, target='subjectivity')
        assert(sequence_bow.shape == (4, 3, 4))


def test_embeddings1():
    # use one word, one class dataset initially
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello'], ['hello']]
    })
    embeddings, Y = uv.embeddings(50, df, target='subjectivity')
    hello = np.array([0.28751, 0.31323, -0.29318, 0.17199, -0.69232,
                      -0.4593, 1.3364, 0.709, 0.12118, 0.11476,
                      -0.48505, -0.088608, -3.0154, -0.54024, -1.326,
                      0.39477, 0.11755, -0.17816, -0.32272, 0.21715,
                      0.043144, -0.43666, -0.55857, -0.47601, -0.095172,
                      0.0031934, 0.1192, -0.23643, 1.3234, -0.45093,
                      -0.65837, -0.13865, 0.22145, -0.35806, 0.20988,
                      0.054894, -0.080322, 0.48942, 0.19206, 0.4556,
                      -1.642, -0.83323, -0.12974, 0.96514, -0.18214,
                      0.37733, -0.19622, -0.12231, -0.10496, 0.45388])
    assert(embeddings[0].all() == hello.all())


def test_embeddings2():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello', 'world'], ['hello', 'world']]
    })
    hello = np.array([0.28751, 0.31323, -0.29318, 0.17199, -0.69232,
                      -0.4593, 1.3364, 0.709, 0.12118, 0.11476,
                      -0.48505, -0.088608, -3.0154, -0.54024, -1.326,
                      0.39477, 0.11755, -0.17816, -0.32272, 0.21715,
                      0.043144, -0.43666, -0.55857, -0.47601, -0.095172,
                      0.0031934, 0.1192, -0.23643, 1.3234, -0.45093,
                      -0.65837, -0.13865, 0.22145, -0.35806, 0.20988,
                      0.054894, -0.080322, 0.48942, 0.19206, 0.4556,
                      -1.642, -0.83323, -0.12974, 0.96514, -0.18214,
                      0.37733, -0.19622, -0.12231, -0.10496, 0.45388])
    world = np.array([0.14968, 0.39511, 0.0060037, 0.30072, 0.36198,
                      0.4877, 0.52916, 0.21741, -0.31404, 0.069926,
                      0.071339, -0.31512, -4.9216, 0.48873, -0.059583,
                      -0.26738, -0.2778, 0.67832, 0.30105, -0.031749,
                      0.36063, -0.47251, -0.22646, 0.12877, 0.24439,
                      0.51575, -0.41108, -0.50409, 0.75757, -0.15384,
                      -0.23331, -1.1831, 0.29457, 0.34829, 0.94679,
                      -0.06382, 0.48357, -0.5668, -0.08089, 0.046397,
                      -1.1265, -0.46418, 0.0087983, 0.018907, -0.049697,
                      0.34804, 0.29426, 0.010255, -0.17227, 0.42294])
    av_embedding = (hello + world) / 2
    embeddings, Y = uv.embeddings(50, df, target='subjectivity')
    assert(embeddings[0].all() == av_embedding.all())


def test_embeddings3():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello'], ['hello']]
    })
    embeddings, Y = uv.embeddings(200, df, target='subjectivity')
    hello = np.array([3.4683e-01, -1.9612e-01, -3.4923e-01, -2.8158e-01, -7.5627e-01,
                      -4.0035e-02, 5.3422e-01, 1.5327e-03, -2.1963e-01, -5.6708e-01,
                      -7.5112e-02, 3.9074e-01, 1.9201e-01, 4.8046e-02, -1.6801e-01,
                      -1.9140e-01, 1.2162e-01, -2.2513e-01, 2.2276e-02, -2.7632e-01,
                      1.0721e-01, -5.8191e-02, -1.7654e-01, -2.0620e-02, -3.9768e-02,
                      1.2619e-01, 1.8927e-01, 1.7017e-01, -2.3453e-02, -4.2349e-01,
                      -4.2640e-02, -2.8101e-01, -3.2461e-01, 3.0870e-01, 9.4529e-02,
                      1.3559e-01, -5.0249e-01, 3.0072e-01, 1.5805e-01, 5.5079e-01,
                      -3.7005e-01, -2.1721e-01, -7.1162e-01, 4.2975e-01, -1.2451e-02,
                      -2.4275e-01, -6.2902e-02, 4.3755e-02, 5.9098e-02, 2.1553e-01,
                      3.4048e-02, -1.5735e-01, -4.4731e-02, -1.2719e-01, 3.3347e-01,
                      2.2386e-01, 3.9716e-01, 8.4382e-02, -4.7057e-02, -1.4943e-01,
                      2.0140e-02, -5.1345e-02, -1.7782e-02, -4.8558e-01, -4.4077e-02,
                      3.8690e-01, -3.5139e-01, 8.8997e-01, 6.6970e-01, -4.4012e-02,
                      4.2673e-01, -1.9671e-01, -5.8553e-02, 1.0207e-01, -3.7026e-01,
                      2.9633e-01, 4.6047e-01, 3.5699e-01, -2.1564e-01, 5.0676e-01,
                      4.0541e-01, 4.1538e-01, 5.3481e-01, 2.2050e-01, 1.5578e-01,
                      -5.7095e-01, -5.5002e-01, 5.3877e-01, 3.3419e-01, -3.3200e-01,
                      -2.0211e-01, -3.7219e-01, -1.1030e-01, 8.9529e-01, -2.1052e-01,
                      -1.3012e-01, -2.4234e-01, -3.0347e-02, 2.2557e-01, 2.4603e-01,
                      -4.7092e-01, 6.5719e-02, -7.6551e-02, -2.3749e-01, -2.7815e-01,
                      2.2050e-01, 2.0567e-01, 5.3484e-01, -1.1766e-01, 8.3034e-02,
                      -5.7123e-02, -1.7614e-01, -4.9715e-01, 1.2829e-01, -1.5242e-01,
                      -7.7388e-01, -7.8140e-01, -4.3172e-01, 6.7606e-01, 2.9269e-01,
                      1.9671e-01, 5.0553e-01, -1.8921e-01, -1.8900e-01, 7.1015e-02,
                      -3.9347e-01, 7.7432e-03, -7.6330e-01, -4.1828e-01, 4.3882e-01,
                      8.9947e-01, -2.4067e-01, 1.3863e-01, 2.5331e-01, -1.0869e-02,
                      -1.0134e-01, -3.4365e-01, 7.1961e-01, 1.6856e-01, 9.6054e-02,
                      -1.7235e-01, -5.2650e-01, 1.9650e-01, -9.1190e-02, -1.7657e-01,
                      1.4870e-01, -2.3176e-02, 9.7574e-01, 7.6538e-01, -2.8794e-01,
                      3.5776e-01, 1.4321e-02, -3.8378e+00, -1.7849e-01, -4.8907e-01,
                      4.2256e-02, -6.9440e-01, -3.7929e-01, -4.3389e-02, -1.5656e-01,
                      7.4036e-01, -3.7037e-01, -3.3502e-01, -5.3957e-02, -1.7478e-01,
                      -6.7377e-02, 4.2054e-01, -5.8659e-02, -2.4218e-01, -8.4078e-02,
                      -3.0372e-01, 1.3549e-01, 2.7088e-01, 4.7949e-01, 3.3393e-02,
                      7.0947e-01, -2.8812e-01, 2.9627e-01, -4.1006e-01, -2.7669e-01,
                      -1.7046e-01, 3.8448e-02, -1.0742e-02, 3.8250e-01, 8.6832e-02,
                      -1.7835e-02, -7.0390e-01, 1.9614e-02, 8.2758e-03, 3.2030e-01,
                      3.5051e-03, 3.3130e-01, 1.5326e-01, -2.2007e-01, -4.5701e-01,
                      -1.7719e-02, -6.1997e-01, -5.2073e-01, 8.2294e-02, -5.4478e-01])
    assert(embeddings[0].all() == hello.all())


def test_embeddings4():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello', 'world'], ['hello', 'world']]
    })
    embeddings, Y = uv.embeddings(200, df, target='subjectivity')
    av_hello_world200 = np.array([1.9130051e-03, 2.1666999e-03, -3.7399994e-04, -3.2306497e-03,
                                  -1.8219000e-03, -2.2650750e-03, 3.2909999e-03, -1.7420864e-03,
                                  2.8809995e-04, -2.8335201e-03, 1.7776900e-03, 2.9458499e-03,
                                  -4.2654499e-03, 1.0300300e-03, 2.5255003e-04, -5.9263001e-04,
                                  8.7807496e-04, -3.2340500e-03, 1.4353650e-04, -3.2029001e-03,
                                  8.2651002e-04, -2.8467050e-03, -5.6722498e-04, 2.0417999e-03,
                                  1.5614500e-04, 4.8985500e-03, 1.3709101e-03, -5.8850000e-04,
                                  -9.6846500e-04, -2.7900499e-03, -4.0468000e-04, -3.2739502e-03,
                                  -6.8880001e-04, 2.6467498e-03, 2.5818450e-03, -1.1332000e-03,
                                  -2.5206283e-03, 6.8950007e-04, 6.2497501e-04, 4.5310003e-03,
                                  -2.2089651e-03, -1.1859500e-03, -4.6040500e-03, 2.5543452e-03,
                                  3.6986449e-03, -7.7481999e-04, 2.3248401e-03, -3.7980250e-03,
                                  1.5753901e-03, 1.7216500e-03, -8.4876001e-04, -1.5659999e-03,
                                  1.9935451e-03, -1.9410999e-03, 1.9978699e-03, -5.5905001e-04,
                                  6.6725002e-04, 3.2052600e-03, 4.3936499e-04, -2.1966000e-03,
                                  5.4153497e-04, -4.7764499e-04, -1.0397100e-03, -2.8805200e-03,
                                  -6.7282497e-04, 3.3663500e-03, -2.5602998e-03, 4.4867536e-03,
                                  4.8792004e-03, -9.9451002e-04, -5.9764995e-04, -7.6354505e-04,
                                  -1.3803650e-03, 3.9565996e-03, 3.7900000e-04, -2.2139000e-03,
                                  4.0794499e-03, 2.3059000e-03, -2.7407501e-03, 3.6869999e-03,
                                  4.2590499e-03, 2.3366250e-03, 5.1531000e-03, 1.8046502e-03,
                                  -9.7699987e-05, -2.8466254e-03, -4.4033499e-03, 3.7326503e-03,
                                  1.0113001e-03, -4.1506998e-03, 1.7000001e-03, -1.5290349e-03,
                                  6.3450006e-04, 5.4739499e-03, -2.3052499e-03, -3.3809000e-03,
                                  -1.0149201e-03, -2.9907348e-03, 3.2613000e-03, -1.6240001e-04,
                                  -3.7742502e-03, 2.7003002e-04, -3.2974549e-03, -3.3500002e-04,
                                  8.2680001e-04, -1.0155500e-03, 1.3388050e-03, 5.4490995e-03,
                                  -1.8390501e-03, 3.2030197e-03, -2.2701651e-03, 1.0091501e-03,
                                  -1.4665499e-03, 2.2417500e-03, -4.7017998e-04, -4.5201499e-03,
                                  -3.3427000e-03, -5.1889499e-03, 2.5007501e-03, 6.5689506e-03,
                                  2.5681001e-03, 1.1977000e-03, 3.1230002e-04, -1.8451500e-03,
                                  -3.3142499e-04, -2.5268998e-03, -1.0441841e-03, -7.0892000e-03,
                                  -1.5630500e-03, 2.4134051e-03, 5.2948999e-03, -1.9439995e-04,
                                  9.3873998e-04, 1.8505999e-03, 2.5721050e-03, -1.7872500e-03,
                                  -5.5831997e-03, 4.2917999e-03, 1.4549500e-03, -2.7448300e-03,
                                  -1.2352050e-03, -1.3524500e-03, 4.6120002e-04, -6.0577499e-04,
                                  -1.0479450e-03, 2.0842999e-03, -1.0313300e-03, 5.6158002e-03,
                                  2.3488500e-03, -1.8074099e-03, 8.8290009e-04, 4.3770499e-04,
                                  -4.4623498e-02, -7.2062499e-04, -3.1301498e-03, -7.4912002e-04,
                                  -1.7857000e-03, -6.7460007e-04, 1.6671051e-03, -2.3018501e-03,
                                  3.1620001e-03, -2.4305501e-03, -3.2261501e-03, -1.7297000e-04,
                                  -1.2623300e-03, -8.5148501e-04, 3.8548498e-03, 2.5905500e-04,
                                  1.0058000e-03, -2.7536401e-03, -1.5267276e-03, 2.2093998e-03,
                                  2.7820501e-03, 1.0326500e-03, 1.8154598e-04, 9.2663504e-03,
                                  -2.3430500e-03, -2.1090001e-04, -2.9712499e-03, -3.1743003e-03,
                                  -6.2583497e-04, -7.8746007e-04, -6.6536001e-04, 1.1015999e-03,
                                  -4.3394975e-05, 1.7230250e-03, -5.1390501e-03, -8.7873003e-04,
                                  -2.9209710e-03, 2.9567000e-03, 2.1834255e-03, 5.0464994e-04,
                                  8.6549000e-04, -2.1865999e-03, -1.4441499e-03, 3.0042552e-03,
                                  -3.0508470e-03, -2.0366001e-03, 5.6049501e-04, -5.8205999e-03])
    assert(embeddings[0].all() == av_hello_world200.all())


def test_embeddings5():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['shizzleshazzle'], ['shizzleshazzle']]  # a word not in the pretrained embeddings
    })
    embeddings, Y = uv.embeddings(50, df, target='subjectivity')
    unk = np.array([-0.21896945, 0.17269082, -0.05617134, 0.06307275, 0.00960663,
                    -0.23461114, -0.16731922, -0.2561416, 0.12990831, -0.34179835,
                    -0.07412009, 0.00533557, 0.70903005, -0.11389928, 0.10613889,
                    0.09186515, 0.15881039, 0.03158559, 0.22414196, 0.20387403,
                    0.05305386, 0.04961181, 0.11807618, -0.10199762, -0.18346203,
                    0.56559574, 0.07183403, 0.04322528, -0.3944287, 0.06828242,
                    0.39541988, 0.08794746, 0.41605816, -0.27820908, -0.51069691,
                    -0.16444368, 0.09734144, 0.02233369, 0.19346113, 0.1591003,
                    0.8865856, -0.01498127, 0.10211429, -0.1295955, -0.32835826,
                    0.13014385, -0.02061018, 0.05735761, 0.14008202, 0.22588644])
    assert(embeddings[0].all() == unk.all())


def test_sequence_embedding1():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello'], ['hello']]
    })
    hello = np.array([0.28751, 0.31323, -0.29318, 0.17199, -0.69232,
                      -0.4593, 1.3364, 0.709, 0.12118, 0.11476,
                      -0.48505, -0.088608, -3.0154, -0.54024, -1.326,
                      0.39477, 0.11755, -0.17816, -0.32272, 0.21715,
                      0.043144, -0.43666, -0.55857, -0.47601, -0.095172,
                      0.0031934, 0.1192, -0.23643, 1.3234, -0.45093,
                      -0.65837, -0.13865, 0.22145, -0.35806, 0.20988,
                      0.054894, -0.080322, 0.48942, 0.19206, 0.4556,
                      -1.642, -0.83323, -0.12974, 0.96514, -0.18214,
                      0.37733, -0.19622, -0.12231, -0.10496, 0.45388])
    embedding_sequences, Y = uv.sequence_embedding(1, 50, df, target='subjectivity')
    assert(np.allclose(embedding_sequences[0][0], hello))


def test_sequence_embedding2():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello', 'world'], ['hello', 'world']]
    })
    world = np.array([0.14968, 0.39511, 0.0060037, 0.30072, 0.36198,
                      0.4877, 0.52916, 0.21741, -0.31404, 0.069926,
                      0.071339, -0.31512, -4.9216, 0.48873, -0.059583,
                      -0.26738, -0.2778, 0.67832, 0.30105, -0.031749,
                      0.36063, -0.47251, -0.22646, 0.12877, 0.24439,
                      0.51575, -0.41108, -0.50409, 0.75757, -0.15384,
                      -0.23331, -1.1831, 0.29457, 0.34829, 0.94679,
                      -0.06382, 0.48357, -0.5668, -0.08089, 0.046397,
                      -1.1265, -0.46418, 0.0087983, 0.018907, -0.049697,
                      0.34804, 0.29426, 0.010255, -0.17227, 0.42294])
    embedding_sequences, Y = uv.sequence_embedding(2, 50, df, target='subjectivity')
    assert(np.allclose(embedding_sequences[0][1], world))


def test_sequence_embedding3():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello', 'world'], ['hello', 'world']]
    })
    hello = np.array([0.28751, 0.31323, -0.29318, 0.17199, -0.69232,
                      -0.4593, 1.3364, 0.709, 0.12118, 0.11476,
                      -0.48505, -0.088608, -3.0154, -0.54024, -1.326,
                      0.39477, 0.11755, -0.17816, -0.32272, 0.21715,
                      0.043144, -0.43666, -0.55857, -0.47601, -0.095172,
                      0.0031934, 0.1192, -0.23643, 1.3234, -0.45093,
                      -0.65837, -0.13865, 0.22145, -0.35806, 0.20988,
                      0.054894, -0.080322, 0.48942, 0.19206, 0.4556,
                      -1.642, -0.83323, -0.12974, 0.96514, -0.18214,
                      0.37733, -0.19622, -0.12231, -0.10496, 0.45388])
    world = np.array([0.14968, 0.39511, 0.0060037, 0.30072, 0.36198,
                      0.4877, 0.52916, 0.21741, -0.31404, 0.069926,
                      0.071339, -0.31512, -4.9216, 0.48873, -0.059583,
                      -0.26738, -0.2778, 0.67832, 0.30105, -0.031749,
                      0.36063, -0.47251, -0.22646, 0.12877, 0.24439,
                      0.51575, -0.41108, -0.50409, 0.75757, -0.15384,
                      -0.23331, -1.1831, 0.29457, 0.34829, 0.94679,
                      -0.06382, 0.48357, -0.5668, -0.08089, 0.046397,
                      -1.1265, -0.46418, 0.0087983, 0.018907, -0.049697,
                      0.34804, 0.29426, 0.010255, -0.17227, 0.42294])
    zero_embedding = np.zeros(50)
    sequence = np.array([hello, world, zero_embedding])
    embedding_sequences, Y = uv.sequence_embedding(3, 50, df, target='subjectivity')
    for i in range(100):
        r = np.random.randint(0, 3)
        assert(np.allclose(embedding_sequences[0][r], sequence[r]))


def test_sequence_embedding4():
    df = pd.DataFrame({
        'subjectivity': [1, 0],
        'polarity': [1, None],
        'tokens': [['hello', 'world'], ['hello', 'world']]
    })
    hello = np.array([0.28751, 0.31323, -0.29318, 0.17199, -0.69232,
                      -0.4593, 1.3364, 0.709, 0.12118, 0.11476,
                      -0.48505, -0.088608, -3.0154, -0.54024, -1.326,
                      0.39477, 0.11755, -0.17816, -0.32272, 0.21715,
                      0.043144, -0.43666, -0.55857, -0.47601, -0.095172,
                      0.0031934, 0.1192, -0.23643, 1.3234, -0.45093,
                      -0.65837, -0.13865, 0.22145, -0.35806, 0.20988,
                      0.054894, -0.080322, 0.48942, 0.19206, 0.4556,
                      -1.642, -0.83323, -0.12974, 0.96514, -0.18214,
                      0.37733, -0.19622, -0.12231, -0.10496, 0.45388])
    world = np.array([0.14968, 0.39511, 0.0060037, 0.30072, 0.36198,
                      0.4877, 0.52916, 0.21741, -0.31404, 0.069926,
                      0.071339, -0.31512, -4.9216, 0.48873, -0.059583,
                      -0.26738, -0.2778, 0.67832, 0.30105, -0.031749,
                      0.36063, -0.47251, -0.22646, 0.12877, 0.24439,
                      0.51575, -0.41108, -0.50409, 0.75757, -0.15384,
                      -0.23331, -1.1831, 0.29457, 0.34829, 0.94679,
                      -0.06382, 0.48357, -0.5668, -0.08089, 0.046397,
                      -1.1265, -0.46418, 0.0087983, 0.018907, -0.049697,
                      0.34804, 0.29426, 0.010255, -0.17227, 0.42294])
    zero_embedding = np.zeros(50)
    sequence = np.array([hello, world, zero_embedding, zero_embedding, zero_embedding])
    embedding_sequences, Y = uv.sequence_embedding(5, 50, df, target='subjectivity')
    for i in range(100):
        r = np.random.randint(0, 3)
        assert(np.allclose(embedding_sequences[0][r], sequence[r]))
