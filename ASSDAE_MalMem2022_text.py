import os
from keras.layers import Input, LayerNormalization, Dropout, GaussianNoise
from lightgbm import LGBMClassifier
from openpyxl import Workbook
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from matplotlib import pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Input, Dense
from tensorflow import keras
from tensorflow.keras import utils
import keras
from keras.layers import Input, Dense
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from datetime import datetime
from tensorflow.keras.layers import Input, Dense, Reshape, MultiHeadAttention
import umap.umap_ as umap
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np
import random
import tensorflow as tf
import pickle
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from keras.callbacks import TensorBoard
import timeit

from ASSDAE import ASSDAE

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 试验过其他shuffle的设置，均无法复现，只有这种shuffle可以复现结果

def compute_metrics(lbl, y_test, y_pred):
    from sklearn.metrics import roc_auc_score
    cm = confusion_matrix(y_test, y_pred)
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    fpr = fp / (fp + tn)
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    acc = (tp + tn) / (tp + fp + fn + tn)
    f1_score = (2 * precision * recall) / (precision + recall)
    # 计算ROC-AUC
    # auc = roc_auc_score(y_test, y_pred)
    print(lbl)
    print('fpr(FAR)=' + str(fpr))
    print('recall=' + str(recall))
    print('precision(DR)=' + str(precision))
    print('accuracy(AC)=' + str(acc))
    print('f1-sore=' + str(f1_score))
    # # 打印ROC-AUC值
    # print("ROC-AUC: {:.4f}".format(auc))
    print(np.array([np.mean(fpr), np.mean(recall), np.mean(recall), np.mean(recall), np.mean(recall)], dtype=np.float32))
    return np.array([np.mean(fpr), np.mean(recall), np.mean(recall), np.mean(recall), np.mean(recall)])

from memory_profiler import profile

@profile
def compute_classifier(lbl, X, y):
    start = timeit.default_timer()

    # 创建XGBoost分类模型
    model = XGBClassifier()

    # 设置十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 初始化列表来存储每个折叠的性能指标
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    fpr_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 训练模型
        model.fit(X_train, y_train)

        # 预测
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        fpr = fp / (fp + tn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        acc = (tp + tn) / (tp + fp + fn + tn)
        f1_score = (2 * precision * recall) / (precision + recall)

        # 将指标添加到列表中
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1_score)
        accuracy_list.append(acc)
        fpr_list.append(fpr)

    # 计算平均值
    mean_precision = np.mean(precision_list)
    mean_recall = np.mean(recall_list)
    mean_f1 = np.mean(f1_list)
    mean_accuracy = np.mean(accuracy_list)
    mean_fpr = np.mean(fpr_list)

    print(lbl)
    # print(f'平均准确率: {mean_accuracy}')
    # print(f'平均召回率: {mean_recall}')
    # print(f'平均精确率: {mean_precision}')
    # print(f'平均F1分数: {mean_f1}')
    # print(f'平均fpr: {mean_fpr}')

    print([mean_fpr, mean_recall, mean_precision, mean_accuracy, mean_f1])

    # compute_metrics(model_name, y_true, y_score)

    end = timeit.default_timer()
    print(f'used time : {end - start}s')
    return [mean_fpr, mean_recall, mean_precision, mean_accuracy, mean_f1]

def preHandle_2(FILE_PATH):
    # 加载数据集
    data = pd.read_csv(FILE_PATH)
    cat_cols = ['Category', 'Class']
    data['Category'] = data['Class']

    # 对分类列进行编码，对数值列进行缩放
    enc = LabelEncoder()
    data[cat_cols] = data[cat_cols].apply(enc.fit_transform)


    scaler = MinMaxScaler()
    cols = data.columns[data.columns != 'Class']
    cols = cols.drop('Category')
    data[cols] = scaler.fit_transform(data[cols])
    random.Random(seed_value).shuffle(data.values)
    print(f"Number of occurrences for each malware class: \n{data['Class'].value_counts()}")
    y = data['Class']
    X = data[cols]
    return X, y


def preHandle_3(FILE_PATH):
    # 加载数据集
    data = pd.read_csv(FILE_PATH)
    data = data[data['Class'] != 'Benign']
    cat_cols = ['Category', 'Class']
    data['Class'] = data['Category'].str.split("-").str[0]
    # 对分类列进行编码，对数值列进行缩放
    enc = LabelEncoder()
    data[cat_cols] = data[cat_cols].apply(enc.fit_transform)


    scaler = MinMaxScaler()
    cols = data.columns[data.columns != 'Class']
    cols = cols.drop('Category')
    data[cols] = scaler.fit_transform(data[cols])
    print(f"Number of occurrences for each malware class: \n{data['Class'].value_counts()}")

    y = data['Class']
    X = data[cols]
    return X, y


def preHandle_15(FILE_PATH):
    # 加载数据集
    data = pd.read_csv(FILE_PATH)
    data = data[data['Class'] != 'Benign']
    cat_cols = ['Category', 'Class']
    data['Class'] = data['Category'].str.split("-").str[1]
    # 对分类列进行编码，对数值列进行缩放
    enc = LabelEncoder()
    data[cat_cols] = data[cat_cols].apply(enc.fit_transform)

    scaler = MinMaxScaler(feature_range=(0, 1))
    cols = data.columns[data.columns != 'Class']
    cols = cols.drop('Category')
    data[cols] = scaler.fit_transform(data[cols])

    data = data.sort_values(by='Class')
    # 使用SMOTE算法对数据进行平衡处理
    X_resampled = data[cols]
    y_resampled = data['Class']
    print(f"Number of occurrences for each malware class after balancing: \n{y_resampled.value_counts()}")
    return X_resampled, y_resampled


def TSNEDimension(X_train, y, savePath, dim):
    X_train_Dim = manifold.TSNE(n_components=dim, perplexity=20, random_state=seed_value).fit_transform(X_train)
    # 保存降维结果为CSV文件
    df = pd.DataFrame(columns=['Dim1', 'Dim2', 'label'])
    df['Dim1'] = X_train_Dim[:, 0]
    df['Dim2'] = X_train_Dim[:, 1]
    print(y)
    df['label'] = y
    df.to_csv(savePath+'X_train_TSNE_4.csv', index=False)
    return X_train_Dim


def UMAPDimension(X_train, y, savePath, dim):
    X_train_Dim = umap.UMAP(n_components=dim, n_neighbors=5, metric='euclidean',random_state=seed_value).fit_transform(X_train)
    # 保存降维结果为CSV文件
    df = pd.DataFrame(columns=['Dim1', 'Dim2', 'label'])
    df['Dim1'] = X_train_Dim[:, 0]
    df['Dim2'] = X_train_Dim[:, 1]
    df['label'] = y
    df.to_csv(savePath+'X_train_UMAP_4.csv', index=False)
    return X_train_Dim


def PCADimension(X_train, y, savePath, dim):
    X_train_Dim = PCA(n_components=dim,random_state=seed_value).fit_transform(X_train)
    # 保存降维结果为CSV文件
    df = pd.DataFrame(columns=['Dim1', 'Dim2', 'label'])
    df['Dim1'] = X_train_Dim[:, 0]
    df['Dim2'] = X_train_Dim[:, 1]
    df['label'] = y
    df.to_csv(savePath+'X_train_PCA_4.csv', index=False)
    return X_train_Dim


@profile
def ASSDAEDimension(X_train, y, hidden_dim, dim):
    savePath = 'demin_MalMem/15/'
    validation_split = 0.2
    spare = 1e-3
    epochs = 30
    batch_size = 256

    # 定义自编码器并编译
    start = timeit.default_timer()
    autoencoder = ASSDAE(X_train.shape[1], dim, hidden_dim, spare)
    autoencoder.name = 'MalMem2022'

    autoencoder.train(X_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    end = timeit.default_timer()

    X_train_Dim = autoencoder.get_encoder_output(X_train)

    # 保存降维结果为CSV文件
    col = [f'Dim{i}' for i in range(dim)] + ['label']
    print(col)
    df = pd.DataFrame(columns=col)
    for i in range(dim):
        df[col[i]] = X_train_Dim[:, i]

    df['label'] = y

    print(df['label'])

    df.to_csv(savePath + 'X_train_ASSDAE_text.csv', index=False)

    print(f'ASSDAEDimension used time : {end - start}s')
    return X_train_Dim, end - start

if __name__ == '__main__':
    FILE_PATH = '../data/CIC-MalMem2022.csv'
    X, y = preHandle_15(FILE_PATH)

    print(X.values)
    dim_range = [4]#range(1, 11)
    # hidden_dim_list = [
    #     [44, 32, 20, 16, 8],
    #     [40, 32, 20, 16],
    #     [40, 32, 16],#
    #     [40, 16],
    #     [40],
    #     [],
    #               ]

    hidden_dim_list = [
        [40, 32, 16],#
                  ]

    spare = 1e-3
    epochs = 30
    batch_size = 256

    X_train = X.values
    y = y.values

    # compute_classifier('ASSDAE:', X_train, y.values) #不适用降维方法的分类效果

    metrics_data = [
        ['隐藏层深度', '第一隐藏层节点数', '编码器输出维度', '时间开销']
    ]
    for hid_Dim in hidden_dim_list:
        for dim in dim_range:
            X_train_encoder, Need_time = ASSDAEDimension(X_train, y, hid_Dim, dim)
            metrics_data.append([len(hid_Dim)] + [hid_Dim[0]] +[dim] + [int(Need_time)])


























