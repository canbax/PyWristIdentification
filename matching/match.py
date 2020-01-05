import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.cross_decomposition import PLSRegression
from sklearn import preprocessing

def build_classifiers(clf_type: str):
    file_path = 'D:\\yusuf\\cs 579\\project\\py-wrist-identifier\\features_set1.npy'
    data = np.load(file_path)
    people = np.unique(data[:, -2])

    for p in people:
        # each person might have 2 wrists
        for i in range(2):
            idx_pos = np.logical_and(data[:, -2] == p, data[:, -1] == i)
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
            E = 1e-6 # prevent divide by zero
            X = (X - x_mu) / (x_sigma + E)
            Y = (Y - y_mu) / (y_sigma + E)
            b = 0
            bias = 0
            
            
            Y = np.ravel(Y)
            if clf_type == 'svm':
                le = preprocessing.LabelEncoder()
                Y = le.fit_transform(Y)
                clf = LinearSVC()
                clf.fit(X, Y)
                bias = clf.intercept_
                b = clf.coef_
            elif clf_type == 'pls':
                pls = PLSRegression(n_components=5)
                pls.fit(X, Y)
                b = pls.coef_
            
            f_name = clf_type + '_classifiers/' + str(int(p)) + '_' + str(i) + '.pkl'
            with open(f_name, 'wb') as f: 
                pickle.dump([b, bias, x_mu, x_sigma, y_mu, y_sigma], f)
                

def match():
    f_name = 'svmclassifiers/1_0.pkl'
    with open(f_name, 'rb') as f:
        b, bias, x_mu, x_sigma, y_mu, y_sigma = pickle.load(f)
        print (b)
        print (bias)
        print (x_mu)                

                
build_classifiers('svm')

