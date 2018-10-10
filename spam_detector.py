import string
from numpy import array
from sklearn.model_selection import KFold
import nltk
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))
vocab = set()
dataset = []
spam = []
spam_dict = {}
ham = []
ham_dict = {}


def read_file(data_dir):
    dataset = open(data_dir).readlines()


def read_data(data_set):
    for line in data_set:
        (first_word, rest) = line.split(maxsplit=1)
        if first_word == 'spam':
            spam.append(clean_text(rest, spam_dict))
        else:
            ham.append(clean_text(rest, ham_dict))
    return spam, ham


def clean_text(text, words_dict):
    """Clean text from stopwords and punctuation"""
    text_words = nltk.word_tokenize(text)
    # text_words = text_words.split()
    result_words = [word for word in text_words if word not in STOP_WORDS and word not in string.punctuation]
    for w in result_words:
        vocab.add(w)
        words_dict[w] = words_dict.get(w, 0) + 1

    result = ' '.join(result_words)

    return result


def clean_new_instance(text):
    voca_intance = set()
    text_words = nltk.word_tokenize(text)
    result_words = [word for word in text_words if word not in STOP_WORDS and word not in string.punctuation]
    for w in result_words:
        voca_intance.add(w)

    return voca_intance


def learn_naive_bayes_mu(data_size, data_documents, data_subset):
    p_dis = {}
    t_j = len(data_documents)
    p_cj = t_j / data_size
    tf_j = sum(data_subset.values())
    for w in vocab:
        tf_ij = data_subset.get(w, 0)
        p_dis[w] = (tf_ij + 1) / (tf_j + data_size)

    return p_cj, p_dis


def classify_naive_bayes(words, p_j, p_dis):
    v_nb = p_j
    for w in words:
        v_nb = v_nb * p_dis[w]
    return v_nb


def classify_kfold(test, errors, p_spam, p_dis_spam, p_ham, p_dis_ham):
    for text in test:
        c = classify_spam(text, p_spam, p_dis_spam, p_ham, p_dis_ham)
    pass


def test_kfold(data):
    k = 0
    j = -1
    errors = []
    while k < 100:
        train = data[k:j]
        test = list(set(data) - set(train))
        (p_spam, p_dis_spam, p_ham, p_dis_ham) = learn_spam(train)
        classify_kfold(test, errors, p_spam, p_dis_spam, p_ham, p_dis_ham)
        k = k + 1
        j = j + 1

    return 1 - (sum(errors) / 100)


def learn_spam(data_set):
    (p_spam, p_dis_spam) = learn_naive_bayes_mu(len(spam) + len(ham), spam, spam_dict)
    (p_ham, p_dis_ham) = learn_naive_bayes_mu(len(spam) + len(ham), ham, ham_dict)
    return p_spam, p_dis_spam, p_ham, p_dis_ham


def classify_spam(text, p_spam, p_dis_spam, p_ham, p_dis_ham):
    words_text = clean_new_instance(text)
    words_text.intersection_update(vocab)
    c_spam = classify_naive_bayes(words_text, p_spam, p_dis_spam)
    c_ham = classify_naive_bayes(words_text, p_ham, p_dis_ham)
    if c_spam > c_ham:
        return 'spam'
    else:
        return 'ham'


def main():
    read_data('dataset/full_dataset')
    accuracy = test_kfold(dataset)
    print()
    # (p_spam, p_dis_spam, p_ham, p_dis_ham) = learn_spam()
    # print(classify_spam(text, p_spam, p_dis_spam, p_ham, p_dis_ham))


if __name__ == '__main__':
    main()
