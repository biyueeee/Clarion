import os
import sys
import re
from datetime import datetime as dt
import pandas as pd
import numpy as np



from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from TrainingFeatures import kmer,GetBaseRatio
from TrainingMetrics import evalu_kfold

from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset, ClassifierChain

import joblib

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
import lightgbm as lgb
from WeightedSeries import WeightedSeries

def read_fasta(inputfile):
    if os.path.exists(inputfile) == False:
        print('Error: file " %s " does not exist.' % inputfile)
        sys.exit(1)
    with open(inputfile) as f:
        record = f.readlines()
    if re.search('>', record[0]) == None:
        print('Error: the input file " %s " must be fasta format!' % inputfile)
        sys.exit(1)

    data = {}

    for line in record:
        if line.startswith('>'):
            name = line.replace('>', '').split('\n')[0]
            data[name] = ''
        else:
            data[name] += line.replace('\n', '')


    return data



def get_label_matrix(label_split):
    label_position = {'Exosome':0,'Nucleus':1,'Nucleoplasm':2,'Chromatin':3,'Cytoplasm':4,'Nucleolus':5, 'Cytosol':6,'Membrane':7, 'Ribosome':8}
    mm = [0,0,0,0,0,0,0,0,0]
    for i in label_split:
        if i in label_position:
            mm[label_position[i]] = 1
    return mm



def read_label(inputfile):
    input = pd.read_csv(inputfile, sep='\t')
    dic = {}
    for A, B, C in zip(input['Gene_ID'], input['Refseq_ID'],input['Annotation_label']):
        CC = C.split('|')
        label_list = get_label_matrix(CC)
        dic[A+'|'+B] = label_list
    return dic





def read_features(inputfile):
    input = pd.read_csv(inputfile, header=None, sep='\t')
    dic = {}
    for A, B in zip(input[0], input[1]):
        BB = B.split(',')
        label_list = [float(i) for i in BB[:-1]]
        dic[A] = label_list
    return dic




def extract_feature(input_seqs, input_labels):
    '''input_seqs is a dictionary: {seq1_name:seq1, seq2_name:seq2, ..., seqn_name:seqn}
       input_labels is also a dictionary: {seq1_name: multi_label1, seq2_name:multi_label2, ..., seqn_name:multi_labeln}
       '''
    print('Now calculating the features of kmer1-6')
    x = []
    y = []
    for i in input_labels:
        seq = input_seqs[i]
        tmp1 = GetBaseRatio(seq)
        tmp2 = kmer(seq, 2)
        tmp3 = kmer(seq, 3)
        tmp4 = kmer(seq, 4)
        tmp5 = kmer(seq, 5)
        tmp6 = kmer(seq, 6)
        tmp = tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6
        x.append(tmp)
        ttmp = input_labels[i]
        y.append(ttmp)

    x = np.array(x)
    y = np.array(y)

    return x, y



def open_fs(filepath):
    ff = open(filepath).readlines()
    indice = [int(i.strip('\n')) for i in ff]
    return indice



def perform_ML(train_x, train_y, val_x, val_y, cc, myweigh):
    '''try my weighted series '''
    from WeightedSeries import WeightedSeries
    clf = WeightedSeries(cc)
    clf.fit(train_x, train_y)
    predic = clf.predict(val_x, weight=myweigh)
    # joblib.dump(clf,model_name)

    acc, one_error, cov, rank_loss, ham_loss, ap = evalu_kfold(val_y, predic)

    return acc, one_error, cov, rank_loss, ham_loss, ap




def perform_BR(train_x, train_y, val_x, val_y, cc, modelname):
    '''input: BR,CC,LP'''
    if modelname == 'BR':
        clf = BinaryRelevance(classifier=cc)
    elif modelname == 'CC':
        clf = ClassifierChain(classifier=cc)
    elif modelname == 'LP':
        clf = LabelPowerset(classifier=cc)
    else:
        clf = BinaryRelevance(classifier=cc)
        print('No specific problem transformation strategy is assigned, now default BR is running... ')

    print('start training', dt.now())
    clf.fit(train_x, train_y)
    print('end training and start predict', dt.now())
    predic = clf.predict(val_x).toarray()

    # joblib.dump(clf, modelname)
    acc, one_error, cov, rank_loss, ham_loss, ap = evalu_kfold(val_y, predic)
    return acc, one_error, cov, rank_loss, ham_loss, ap





def fold_10_cross_valication(x, y, clf, myweight):
    ACC = []
    ONE_E = []
    COVER = []
    RANKLOSS = []
    HAMLOSS = []
    AP = []


    kf = KFold(n_splits=10, shuffle=True, random_state=6)
    for train_index, test_index in kf.split(x):
        train_x, train_y = x[train_index], y[train_index]
        val_x, val_y = x[test_index], y[test_index]
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(val_x)
        test_y = np.array(val_y)

        ac, one, cov, rankloss, hamloss, ap = perform_ML(train_x, train_y, test_x, test_y, clf, myweight)
        # ac, one, cov, rankloss, hamloss, ap = perform_BR(train_x,train_y, test_x, test_y, clf, 'BR')
        ACC.append(ac)
        ONE_E.append(one)
        COVER.append(cov)
        RANKLOSS.append(rankloss)
        HAMLOSS.append(hamloss)
        AP.append(ap)

    print('end training and predict', dt.now())
    print('10-fold ACC', ACC)
    print('10-fold One-error', ONE_E)
    print('10-fold Coverage', COVER)
    print('10-fold RankingLoss', RANKLOSS)
    print('10-fold HammingLoss', HAMLOSS)
    print('10-fold AveragePrecision', AP)


    print('10-fold ACC', np.mean(ACC))
    print('10-fold One-error', np.mean(ONE_E))
    print('10-fold Coverage', np.mean(COVER))
    print('10-fold RankingLoss', np.mean(RANKLOSS))
    print('10-fold HammingLoss', np.mean(HAMLOSS))
    print('10-fold AveragePrecision', np.mean(AP))
    print()





def perform_train():
    '''training process'''
    id_train = read_fasta('training_validation_seqs')
    id_train_y = read_label('training_validation_labels')

    train_x, train_y = extract_feature(id_train, id_train_y)

    # comparison with other machine learning algorithms
    weight_set = [0.25, 0.5, 0.75]
    for i in weight_set:
        print('XGBoost when weight=', i)
        xgb = XGBClassifier(use_label_encoder=False)#tree_method='gpu_hist',gpu_id=2,use_label_encoder=False
        fold_10_cross_valication(train_x, train_y, xgb, i)

        print('CatBoost when weight=', i)
        cat = CatBoostClassifier(silent=True)
        fold_10_cross_valication(train_x,train_y,cat, i)

        print('KNN when weight=', i)
        knn = KNeighborsClassifier()
        fold_10_cross_valication(train_x, train_y, knn, i)

        print('LR when weight=', i)
        logis = LogisticRegression()
        fold_10_cross_valication(train_x, train_y, logis, i)

        print('RF when weight=', i)
        rf = RandomForestClassifier()
        fold_10_cross_valication(train_x, train_y, rf, i)

        print('LightGBM when weight=', i)
        gbm = lgb.LGBMClassifier()
        fold_10_cross_valication(train_x, train_y, gbm, i)

        print('MLP when weight=', i)
        mlp = MLPClassifier()
        fold_10_cross_valication(train_x, train_y, mlp, i)




    # comparison with BR,CC and LP
    xgb = XGBClassifier(use_label_encoder=False)
    fold_10_cross_valication(train_x, train_y, xgb, 0)








def perform_test():
    '''independent test'''
    id_test = read_fasta('independent_seqs')
    id_test_y = read_label('independent_labels')
    test_x, test_y = extract_feature(id_test, id_test_y)



    clf = joblib.load('model')

    # calculate multi-label performance
    predic = clf.predict(test_x,weight=0.65)
    acc_ex, one_error, cov, rank_loss, ham_loss, ap = evalu_kfold(test_y, predic)
    print('independent test ACC_ex', acc_ex)
    print('independent test One-error', one_error)
    print('independent test Coverage', cov)
    print('independent test Ranking Loss', rank_loss)
    print('independent test Hamming Loss', ham_loss)
    print('independent test AveragePrecision', ap)



    # calculate single-label performance
    from sklearn.metrics import accuracy_score
    localization_set = {0:'Exosome', 1:'Nucleus', 2:'Nucleoplasm',
                        3:'Chromatin', 4:'Cytoplasm', 5:'Nucleolus',
                        6:'Cytosol', 7:'Membrane', 8:'Ribosome'}
    for i in range(9):
        acc = accuracy_score(test_y[:,i],predic[:,i])
        print(localization_set[i],'Accuracy is',acc)










# perform_train()
perform_test()






