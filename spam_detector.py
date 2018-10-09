import nltk
from nltk.corpus import stopwords
import string


def read_data(data_dir):
    spam = []
    ham = []
    with open(data_dir, 'r') as data_set:
        for line in data_set:
            (first_word, rest) = line.split(maxsplit=1)
            if first_word == 'spam':
                spam.append(rest)
            else:
                ham.append(rest)

    return spam, ham


def clean_data(data):
    tokens = [nltk.word_tokenize(d) for d in data]
    words = set()
    for t in tokens:
        words = words.union(clean_words(t))
    return words


def clean_words(words):
    stop = set(stopwords.words('english'))
    word_count = {}
    for w in words:
        if w not in stop and w not in string.punctuation:
            word_count[w.lower()] = word_count[w.lower()] + 1
    return word_count


def clean_words_set(words):
    stop = set(stopwords.words('english'))
    words_cleaned = set()
    for w in words:
        if w not in stop and w not in string.punctuation:
            words_cleaned.add(w.lower())
    return words_cleaned


def learn_naive_bayes_be(data_size, data_subset, vocab, words):
    p_dis = {}
    t_j = len(data_subset)
    p_j = t_j / data_size
    for w in vocab:
        p_dis[w] = (words[w] + 1) / (t_j + 2)

    return p_j, p_dis


def classify_naive_bayes(words, p_j, p_dis):
    v_nb = p_j
    for w in words:
        v_nb = v_nb * p_dis[w]
    return v_nb


def learn_spam(filename):
    spam, ham = read_data(filename)
    words_spam = clean_data(spam)
    words_ham = clean_data(ham)
    vocab = words_ham.union(words_spam)
    data_set_size = len(spam) + len(ham)
    (p_spam, p_dis_spam) = learn_naive_bayes_be(data_set_size, spam, vocab, words_spam)
    (p_ham, p_dis_ham) = learn_naive_bayes_be(data_set_size, ham, vocab, words_ham)
    return p_spam, p_dis_spam, p_ham, p_dis_ham, vocab


def classify_spam(text, vocab, p_spam, p_dis_spam, p_ham, p_dis_ham):
    words_text = clean_words_set(nltk.word_tokenize(text))
    words_text.intersection_update(vocab)
    c_spam = classify_naive_bayes(words_text, p_spam, p_dis_spam)
    c_ham = classify_naive_bayes(words_text, p_ham, p_dis_ham)
    print('c_ham: ', c_ham)
    print('c_spam: ', c_spam)
    if c_spam > c_ham:
        return 'spam'
    else:
        return 'ham'


def main():
    text = 'URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010 T&C www.dbuk.net LCCLTD POBOX 4403LDNW1A7RW18'
    (p_spam, p_dis_spam, p_ham, p_dis_ham, vocab) = learn_spam('dataset/full_dataset')
    print(classify_spam(text, vocab, p_spam, p_dis_spam, p_ham, p_dis_ham))


if __name__ == '__main__':
    main()
