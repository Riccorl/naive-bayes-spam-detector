import spam_detector as sd


def classify_kfold(test, p_spam, p_dis_spam, p_ham, p_dis_ham):
    error, t_p, t_n, f_p, f_n = 0, 0, 0, 0, 0

    for text in test:
        true_answer = text[0]
        rest = text[1]
        guess_answer = sd.classify_spam(rest, p_spam, p_dis_spam, p_ham, p_dis_ham)
        if guess_answer != true_answer:
            error = error + 1
            if true_answer == 'ham':
                f_n = f_n + 1
            else:
                f_p = f_p + 1
        else:
            if true_answer == 'ham':
                t_p = t_p + 1
            else:
                t_n = t_n + 1

    return error / len(test), t_p, t_n, f_p, f_n


def test_portion(data, i, portion):
    test = data[:i] + data[i + portion:]
    train = data[i + 1:i + portion - 1]
    return test, train


def test_kfold(vocab, data, k):
    i, t_p, t_n, f_p, f_n = 0, 0, 0, 0, 0
    portion_len = len(data) // k
    errors = []

    while i < k:
        (train, test) = test_portion(data, i, portion_len)
        (p_spam, p_dis_spam, p_ham, p_dis_ham) = sd.learn_spam(vocab, train)
        (error, t_p, t_n, f_p, f_n) = classify_kfold(test, p_spam, p_dis_spam, p_ham, p_dis_ham)
        errors.append(error)
        print('test %d, error: %f' % (i + 1, error))
        i = i + 1

    accuracy = 1 - (sum(errors) / k)
    precision = t_p / (t_p + f_n)
    recall = t_p / (t_p + f_p)
    f_score = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, f_score
