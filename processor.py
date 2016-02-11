# author = 'yanhe'

import glob
import math
from operator import itemgetter


def lang_model(l0, l1, l2, l3, test_path, train_path):
    print 'Start calculating perplexity score.'
    train_content = ''
    train_file = open(train_path, 'r')
    for line in train_file:
        train_content += line
    train_vocab, tokens = get_vocab(train_content.split())
    vocab_size = len(train_vocab)
    model_list = []
    model_size = []
    uniform_model = dict((k, 1) for k in train_vocab)
    model_list.append(uniform_model)
    model_size.append(vocab_size)
    for i in xrange(3):
        cur_model, cur_size = ngram_model(train_path, tokens, i+1)
        model_list.append(cur_model)
        model_size.append(cur_size)

    test_content = ''
    test_file = open(test_path, 'r')
    for line in test_file:
        test_content += line
    test_tokens = test_content.split()
    for i in xrange(2):
        test_tokens.insert(0, '<start>')

    s = 0.0
    n = 0
    for i in xrange(2, len(test_tokens)):
        if test_tokens[i] not in train_vocab:
            test_tokens[i] = 'UNKNOWNWORD'
        idx = [i-2, i-1, i]
        tri = itemgetter(*idx)(test_tokens)
        lq = get_score(tri, model_list, model_size, l0, l1, l2, l3)
        s += lq
        n += 1
    print math.exp(-s / n)


def get_score(tri, model_list, model_size, l0, l1, l2, l3):
    pp0 = l0 * model_list[0].get(tri[2:3][0], 0) / model_size[0]
    pp1 = l1 * model_list[1].get(tri[2:3][0], 0)/ model_size[1]
    if model_list[1].get(tri[1:2][0], 0) == 0:
        pp2 = 0
    else:
        pp2 = l2 * model_list[2].get(tri[1:3], 0) / model_list[1].get(tri[1:2][0], 0)
    if model_list[2].get(tri[0:2], 0) == 0:
        pp3 = 0
    else:
        pp3 = l3 * model_list[3].get(tri[0:3], 0) / model_list[2].get(tri[0:2], 0)
    pp_score = math.log(pp0 + pp1 + pp2 + pp3)
    return pp_score


def get_vocab(tokens):
    vocab_dict = {}
    vocab_set = set()
    for token in tokens:
        freq = vocab_dict.get(token, 0) + 1
        vocab_dict[token] = freq
    for i in xrange(len(tokens)):
        if vocab_dict[tokens[i]] < 5:
            tokens[i] = 'UNKNOWNWORD'
            vocab_dict[tokens[i]] = -1
            vocab_set.add('UNKNOWNWORD')
        else:
            vocab_set.add(tokens[i])
    return vocab_set, tokens


def ngram_model(file, tokens, n):
    for i in xrange(n - 1):
        tokens.insert(0, '<start>')
    ngram_dict = {}
    if n > 1:
        ngrams = find_ngram(tokens, n)
    else:
        ngrams = tokens
    ngram_size = len(ngrams)
    for ngram in ngrams:
        freq = ngram_dict.get(ngram, 0) + 1
        ngram_dict[ngram] = freq
    return ngram_dict, ngram_size


def find_ngram(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


if __name__ == "__main__":
    train = 'train/games.txt'
    test = 'train/health.txt'
    lang_model(0.25, 0.25, 0.25, 0.25, test, train)