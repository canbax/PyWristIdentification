import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression
from sklearn import preprocessing
import time
from os import listdir
import matplotlib.pyplot as plt
from scipy.stats import weibull_min


def build_classifiers(set_name: str, clf_type: str, clf_suffix=''):
    file_path = 'results/features_lbp' + set_name + '.npy'
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


def get_response_and_ranking(num_probe, clf_file_name, probe_x, probe_y, zero_betas=[]):

    with open(clf_file_name, 'rb') as f:
        betas, biases, clf_y, x_mu, x_sigma = pickle.load(f)

    num_clf = len(clf_y)
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

    return resp_vec, s, clf_y


def get_CMC(num_probe, ranking, max_rank=30):
    rankPP = np.zeros((num_probe, 1))
    for i in range(num_probe):
        if len(ranking[i]) == 0:
            rankPP[i] = 0
        else:
            # python uses zero indexing, rank them from 1
            rankPP[i] = ranking[i][0] + 1

    cmc = np.zeros((max_rank, 1))
    for i in range(1, max_rank):
        cmc[i] = len(rankPP[rankPP <= i]) / len(rankPP) * 100

    return cmc


def match(galery_set, probe_set, clf_type, clf_id: str, clf_suffix='', cnt=0, zero_betas=[], legend_str=''):

    probe_file_path = 'results/features' + clf_suffix + probe_set + '.npy'
    probe_data = np.load(probe_file_path)
    # shuffle probe data
    probe_data = np.random.permutation(probe_data)
    probe_x = probe_data[:, :-1]
    probe_y = probe_data[:, -1]

    num_probe, _ = probe_x.shape

    clf_path = 'results/' + galery_set + '_' + clf_type + '_clf/'
    # read all classifiers from 1 file
    clf_file_name = clf_path + 'clf' + clf_id + clf_suffix + '.pkl'
    _, s, _ = get_response_and_ranking(
        num_probe, clf_file_name, probe_x, probe_y, zero_betas)

    max_rank = 30
    cmc = get_CMC(num_probe, s, max_rank)

    clf_str = galery_set + ' ' + clf_type + ' ' + \
        clf_suffix + ' ' + clf_id + ' ' + legend_str
    plt.plot(np.arange(max_rank)[1:max_rank],
             cmc[1:max_rank, 0], label=clf_str, linestyle=['-', '--', '-.', ':'][cnt], linewidth=8)


def meta_match(line_style=0, legend_str=''):
    n_outlier = 1
    tail_factor = .2  # fraction of the gallery size that will determine tail's length

    p1 = 'results/SET1'
    clf_paths = [p1 + '_pls_clf/clf5.pkl', p1 + '_svm_clf/clf1000.pkl',
                 p1 + 'p_pls_clf/clf5.pkl', p1 + 'p_svm_clf/clf1000.pkl']
    p2 = 'results/featuresSET2'
    probe_paths = [p2 + '.npy', p2 + '.npy', p2 + 'p.npy', p2 + 'p.npy']

    responses = []
    clf_ys = []
    probe_ys = []  # labels should be the same for all representations
    for i, clf_path in enumerate(clf_paths):
        probe_data = np.load(probe_paths[i])
        # shuffle probe data
        probe_data = np.random.permutation(probe_data)
        probe_x = probe_data[:, :-1]
        probe_y = probe_data[:, -1]
        probe_ys.append(probe_y)
        r, _, clf_y = get_response_and_ranking(
            probe_x.shape[0], clf_path, probe_x, probe_y)
        responses.append(r)
        clf_ys.append(clf_y)

    num_probe = responses[0].shape[0]
    # fraction of number of classifers AS STATED IN PAPER, NOT IN CODES
    n2 = int(round(responses[0].shape[1] * tail_factor))
    ranking = []

    for i in range(num_probe):
        d1 = responses[0][i, :] + np.abs(np.min(responses[0][i, :]))
        d2 = responses[1][i, :] + np.abs(np.min(responses[1][i, :]))
        d3 = responses[2][i, :] + np.abs(np.min(responses[2][i, :]))
        d4 = responses[3][i, :] + np.abs(np.min(responses[3][i, :]))

        idx1 = np.argsort(d1)[::-1]
        d1 = d1[idx1]
        idx2 = np.argsort(d2)[::-1]
        d2 = d2[idx2]
        idx3 = np.argsort(d3)[::-1]
        d3 = d3[idx3]
        idx4 = np.argsort(d4)[::-1]
        d4 = d4[idx4]

        p1 = weibull_min.fit(d1[n_outlier:n2, ])
        p2 = weibull_min.fit(d2[n_outlier:n2, ])
        p3 = weibull_min.fit(d3[n_outlier:n2, ])
        p4 = weibull_min.fit(d4[n_outlier:n2, ])

        c1 = weibull_min.cdf(d1[0], p1[0], p1[1], p1[2])
        c2 = weibull_min.cdf(d2[0], p2[0], p2[1], p2[2])
        c3 = weibull_min.cdf(d3[0], p3[0], p3[1], p3[2])
        c4 = weibull_min.cdf(d4[0], p4[0], p4[1], p4[2])

        idx_RS = np.argmax([c1, c2, c3, c4])

        # ordering = np.argsort(resp)[::-1]
        # s.append(np.nonzero(probe_y[i] == clf_y[ordering])[0])
        idx_arr = [idx1, idx2, idx3, idx4]
        ranking.append(np.nonzero(
            probe_ys[idx_RS][i] == clf_ys[idx_RS][idx_arr[idx_RS]])[0])

    cmc = get_CMC(num_probe, ranking)
    print(cmc)


def plot_meta_match():
    cmc = np.array([0, 
                    81.88705234, 
                    85.60606061,
                    86.84573003,
                    87.67217631,
                    88.49862259,
                    89.1184573,
                    89.46280992,
                    90.08264463,
                    90.42699725,
                    90.84022039,
                    91.25344353,
                    91.4600551,
                    91.52892562,
                    91.87327824,
                    92.07988981,
                    92.21763085,
                    92.56198347,
                    92.76859504,
                    92.83746556,
                    92.83746556,
                    92.97520661,
                    93.18181818,
                    93.18181818,
                    93.25068871,
                    93.31955923,
                    93.45730028,
                    93.5261708,
                    93.5261708,
                    93.66391185])

    max_rank = 30
    plt.plot(np.arange(max_rank)[1:max_rank],
             cmc[1:max_rank], label='meta match with EVT', linestyle='-', linewidth=8)

t = time.time()
# build_classifiers('SET1p', 'svm', 'lbp')

font_size = 32
plt.figure()
plt.title('meta match vs best matcher', fontsize=font_size)
plt.tick_params(labelsize=font_size)
plt.xlabel('rank', fontsize=font_size)
plt.ylabel('rank-m identification rate (%)', fontsize=font_size)

# match('SET1', 'SET2', 'pls', '5', '', 0, [], '')
# match('SET1p', 'SET2p', 'svm', '1000', 'lbp', 1, [], 'LBP features')
meta_match()
# plot_meta_match()

legend = plt.legend(fontsize=font_size*3/4)


plt.show()

print(time.time() - t)
