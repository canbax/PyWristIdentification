import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression
from sklearn import preprocessing
import time
from os import listdir


def build_classifiers(set_name: str, clf_type: str):
    file_path = 'D:\\yusuf\\cs 579\\term project\\py-wrist-indentifier\\features_' + \
        set_name + '.npy'
    data = np.load(file_path)
    people = np.unique(data[:, -1])

    for p in people[0:1]:
        idx_pos = data[:, -1] == p
        idx_neg = np.logical_not(idx_pos)

        if idx_pos.shape[0] < 1:
            break
        x_pos = data[idx_pos, :-2]
        x_neg = data[idx_neg, :-2]

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
            clf = LinearSVC(max_iter=1000)
            clf.fit(X, Y)
            bias = clf.intercept_
            b = clf.coef_
        elif clf_type == 'pls':
            pls = PLSRegression(n_components=5)
            pls.fit(X, Y)
            b = pls.coef_

        f_name = set_name + '_' + clf_type + '_clf/' + str(int(p)) + '.pkl'
        with open(f_name, 'wb') as f:
            pickle.dump([b, bias, x_mu, x_sigma, y_mu, y_sigma], f)


def match(galery_set, probe_set, clf_type):

    probe_file_path = 'features_' + probe_set + '.npy'
    probe_data = np.load(probe_file_path)
    # shuffle probe data
    probe_data = np.random.permutation(probe_data)
    probe_x = probe_data[:, :-1]
    probe_y = probe_data[:, -1]

    num_probe, num_feature = probe_x.shape

    clf_path = galery_set + '_' + clf_type + '_clf/'
    clf_files = listdir(clf_path)
    num_clf = len(clf_files)

    # vectorized variables for classifiers
    betas = np.zeros((num_clf, num_feature))
    biases = np.zeros((num_clf, 1))
    x_mus = np.zeros((num_clf, num_feature))
    y_mus = np.zeros((num_clf, 1))
    x_sigmas = np.zeros((num_clf, num_feature))
    y_sigmas = np.zeros((num_clf, 1))
    clf_y = np.zeros((num_clf, 1))

    for i, clf_file in enumerate(clf_files):
        with open(clf_file, 'rb') as f:
            b, bias, x_mu, x_sigma, y_mu, y_sigma = pickle.load(f)
            betas[i, :] = b
            biases[i] = bias
            x_mus[i, :] = x_mu
            y_mus[i] = y_mu
            x_sigmas[i, :] = x_sigma
            y_sigmas[i] = y_sigma
            clf_y[i] = int(clf_file[:-4])

    resp_vec = np.zeros((num_probe, num_clf))
    s = []
    for i in range(num_probe):
        x_i = (probe_x[i, :] - x_mus[i, :]) / x_sigmas[i, :]
        resp = betas * x_i.T
        resp_vec[i, :] = resp
        idx = np.argsort(resp[::-1])
        s.append(np.nonzero(probe_y[i] == np.unique(clf_y[idx])))

    rankPP = np.zeros((1, num_probe))
    for i in range(num_probe):
        if len(s[i]) == 0:
            rankPP[i] = 0
        else:
            rankPP[i] = s[0]

    cmc = np.zeros((1, num_probe))
    for i in range(num_probe):
        cmc[i] = len(rankPP[rankPP <= i]) / len(rankPP)


t = time.time()
build_classifiers('SET1', 'svm')
print(t - time.time())
