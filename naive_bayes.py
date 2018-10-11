def learn_naive_bayes_mu(vocab, data, docs_size, dict_class):
    p_dis = {}
    t_j = docs_size
    p_cj = t_j / len(data)
    tf_j = sum(dict_class.values())
    for w in vocab:
        tf_ij = dict_class.get(w, 0)
        p_dis[w] = (tf_ij + 1) / (tf_j + len(vocab))

    return p_cj, p_dis


def classify_naive_bayes(words, p_j, p_dis):
    v_nb = p_j
    for w in words:
        v_nb = v_nb * p_dis[w]
    return v_nb
