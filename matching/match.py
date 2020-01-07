import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression
from sklearn import preprocessing
import time
from os import listdir
import matplotlib.pyplot as plt


def build_classifiers(set_name: str, clf_type: str, clf_suffix=''):
    file_path = 'results/features' + set_name + '.npy'
    data = np.load(file_path)
    people = np.unique(data[:, -1])

    bs = []
    biases = []
    clf_labels = []
    x_mu = 0
    x_sigma = 0
    num_iter_4_svm = 1000
    num_comp_4_pls = 5
    for p in people:
        idx_pos = data[:, -1] == p
        idx_neg = np.logical_not(idx_pos)

        if idx_pos.shape[0] < 1:
            break
        x_pos = data[idx_pos, :-1]
        x_neg = data[idx_neg, :-1]

        X = np.vstack((x_pos, x_neg))
        Y = np.vstack(
            (np.ones((x_pos.shape[0], 1)), np.ones((x_neg.shape[0], 1)) * -1))

        x_mu = np.mean(X, axis=0)
        y_mu = np.mean(Y, axis=0)
        x_sigma = np.std(X, axis=0)
        y_sigma = np.std(Y, axis=0)
        E = 1e-6  # prevent divide by zero
        X = (X - x_mu) / (x_sigma + E)
        Y = (Y - y_mu) / (y_sigma + E)
        b = 0
        bias = 0

        Y = np.ravel(Y)
        if clf_type == 'svm':
            le = preprocessing.LabelEncoder()
            Y = le.fit_transform(Y)
            clf = LinearSVC(max_iter=num_iter_4_svm, class_weight='balanced')
            clf.fit(X, Y)
            bias = clf.intercept_
            b = clf.coef_
        elif clf_type == 'pls':
            pls = PLSRegression(n_components=num_comp_4_pls)
            pls.fit(X, Y)
            b = pls.coef_

        bs.append(np.ravel(b))
        biases.append(np.ravel(bias))
        clf_labels.append(int(p))

    bs = np.array(bs)
    biases = np.array(biases)
    clf_labels = np.array(clf_labels)

    i = num_iter_4_svm
    if clf_type == 'pls':
        i = num_comp_4_pls
    i = str(i)
    f_name = 'results/' + set_name + '_' + \
        clf_type + '_clf/clf' + i + clf_suffix + '.pkl'
    with open(f_name, 'wb') as f:
        pickle.dump([bs, biases, clf_labels, x_mu, x_sigma], f)


def match(galery_set, probe_set, clf_type, clf_id: str, clf_suffix='', cnt=0, zero_betas=[], legend_str=''):

    probe_file_path = 'results/features' + probe_set + '.npy'
    probe_data = np.load(probe_file_path)
    # shuffle probe data
    probe_data = np.random.permutation(probe_data)
    probe_x = probe_data[:, :-1]
    probe_y = probe_data[:, -1]

    num_probe, _ = probe_x.shape

    clf_path = 'results/' + galery_set + '_' + clf_type + '_clf/'
    # read all classifiers from 1 file
    with open(clf_path + 'clf' + clf_id + clf_suffix + '.pkl', 'rb') as f:
        betas, biases, clf_y, x_mu, x_sigma = pickle.load(f)

    num_clf = len(clf_y)

    # It should already have unique elements but anyway
    # clf_y = np.unique(clf_y)

    resp_vec = np.zeros((num_probe, num_clf))
    s = []
    if len(zero_betas) > 0:
        betas[:, zero_betas] = 0
        
    for i in range(num_probe):
        x_i = (probe_x[i, :] - x_mu) / (x_sigma + 1e-6)
        resp = np.matmul(betas, x_i.T)
        resp = resp + biases.reshape(num_clf, )
        resp_vec[i, :] = resp
        ordering = np.argsort(resp)[::-1]
        s.append(np.nonzero(probe_y[i] == clf_y[ordering])[0])

    rankPP = np.zeros((num_probe, 1))
    for i in range(num_probe):
        if len(s[i]) == 0:
            rankPP[i] = 0
        else:
            # python uses zero indexing, rank them from 1
            rankPP[i] = s[i][0] + 1

    max_rank = 30
    cmc = np.zeros((max_rank, 1))
    for i in range(1, max_rank):
        cmc[i] = len(rankPP[rankPP <= i]) / len(rankPP) * 100
    print(cmc.shape)

    clf_str = galery_set + ' ' + clf_type + ' ' + clf_suffix + ' ' + clf_id + ' ' + legend_str
    plt.plot(np.arange(max_rank)[1:max_rank],
             cmc[1:max_rank, 0], label=clf_str, linestyle=['-', '--', '-.', ':'][cnt], linewidth=8)


t = time.time()
# build_classifiers('SET1', 'svm', 'balanced')

font_size = 32
plt.figure()
plt.title('Effectiveness of features on PLS', fontsize=font_size)
plt.tick_params(labelsize=font_size)
plt.xlabel('rank', fontsize=font_size)
plt.ylabel('rank-m identification rate (%)', fontsize=font_size)

match('SET1', 'SET2', 'pls', '5', '', 0, np.arange(13074), 'Gabon features')
match('SET1', 'SET2', 'pls', '5', '', 1, np.arange(13074, 15186), 'LBP features')
match('SET1', 'SET2', 'pls', '5', '', 2, [], 'All features')

legend = plt.legend(fontsize=font_size*3/4)

plt.show()

print(time.time() - t)
