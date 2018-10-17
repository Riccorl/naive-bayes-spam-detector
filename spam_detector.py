import string

import nltk
from nltk.corpus import stopwords

import eval_train
import naive_bayes as bayes

STOP_WORDS = set(stopwords.words('english'))
vocab = set()
data = []
counter_spam, counter_ham = 0, 0
dict_spam, dict_ham = {}, {}


def clean_text(text):
    tokens = nltk.word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    return [w for w in words if w not in STOP_WORDS]


def read_data(data_set):
    global counter_spam, counter_ham
    for line in data_set:
        (answer, text) = line.split(maxsplit=1)
        if answer == 'spam':
            counter_spam = counter_spam = 1
        else:
            counter_ham = counter_ham + 1
        data.append([answer, prepare_text(answer, text)])

    return data


def prepare_text(answer, text):
    words = clean_text(text)
    for w in words:
        vocab.add(w)
        if answer == 'spam':
            dict_spam[w] = dict_spam.get(w, 0) + 1
        else:
            dict_ham[w] = dict_ham.get(w, 0) + 1

    return ' '.join(words)


def prepare_new_instance(text):
    vocab_intance = set()
    words = clean_text(text)
    for w in words:
        vocab_intance.add(w)

    return vocab_intance


def learn_spam(vocab, data_set):
    (p_spam, p_dis_spam) = bayes.learn_multinomial(vocab, data_set, counter_spam, dict_spam)
    (p_ham, p_dis_ham) = bayes.learn_multinomial(vocab, data_set, counter_ham, dict_ham)
    return p_spam, p_dis_spam, p_ham, p_dis_ham


def classify_spam(text, p_spam, p_dis_spam, p_ham, p_dis_ham):
    words_text = prepare_new_instance(text)
    words_text.intersection_update(vocab)
    c_spam = bayes.classify_naive_bayes(words_text, p_spam, p_dis_spam)
    c_ham = bayes.classify_naive_bayes(words_text, p_ham, p_dis_ham)
    if c_spam > c_ham:
        return 'spam'
    else:
        return 'ham'


def main():
    file = open('dataset/full_dataset').readlines()
    data_set = read_data(file)
    accuracy, precision, recall, f_score = eval_train.test_kfold(vocab, data_set, 10)
    print('Accuracy: %f \nPrecision: %f\nRecall: %f\nF-score: %f \n'
          % (accuracy, precision, recall, f_score))
    # (p_spam, p_dis_spam, p_ham, p_dis_ham) = learn_spam()
    # print(classify_spam(text, p_spam, p_dis_spam, p_ham, p_dis_ham))


if __name__ == '__main__':
    main()
