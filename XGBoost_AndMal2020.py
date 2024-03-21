import itertools
import os
import timeit

from keras.layers import Input, LayerNormalization, Dropout, GaussianNoise
from lightgbm import LGBMClassifier
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
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
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np
import random
import tensorflow as tf
import pickle
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from keras.callbacks import TensorBoard
from openpyxl import Workbook
from ASSDAE import ASSDAE
from BoostForest import BoostTreeClassifier, BoostForestClassifier

from line_profiler import profile, LineProfiler


# model = BoostForestClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1, n_estimators=50)
# model = BoostTreeClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1)

# def plot_confusion_matrix(y_true, y_pred, num_classes, num):
#     import seaborn as sns
#
#     classes = ['Adware', 'Backdoor', 'FileInfector', 'No_Category', 'PUA', 'Ransomware',
#                'Riskware', 'Scareware', 'Trojan', 'Trojan_Banker', 'Trojan_Dropper', 'Trojan_SMS', 'Trojan_Spy', 'Zero_Day'] if num_classes > 2 else ['begine', 'malware']
#
#     title = "CCCS-CIC-AndMal-2020 Family Classification" if num_classes > 2 else ['CCCS-CIC-AndMal-2020 Binary Classification']
#
#     cm = confusion_matrix(y_true, y_pred)
#     cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
#
#     plt.figure(figsize=(12, 12.5))
#     sns.heatmap(cm_normalized, annot=True, cmap="Blues", xticklabels=classes, yticklabels=classes, fmt=".1f")
#
#     plt.title(title)
#     plt.xlabel('Predicted')
#     plt.ylabel('True')
#
#
#     # 添加刻度标签
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=60)
#     plt.yticks(tick_marks, classes)
#
#     for i in range(cm_normalized.shape[0]):
#         for j in range(cm_normalized.shape[1]):
#             plt.text(j + 0.5, i + 0.5, format(cm_normalized[i, j], '.1f'), horizontalalignment="center", color="white")
#
#     plt.show()
#
#
#
#     # # 保存图像，并设置dpi
#     # if not os.path.exists('./confusionPlt/CCCS-CIC-AndMal-2020/'):
#     #     os.makedirs('./confusionPlt/CCCS-CIC-AndMal-2020/')
#     # plt.savefig(f"./confusionPlt/CCCS-CIC-AndMal-2020/{title}_{num}.png", dpi=300)
#     # plt.show()
#

def plot_roc_curves(y_true, y_pred, num_classes, num):
    # 初始化真正例率（TPR）、假正例率（FPR）和AUC
    tprs = []
    fprs = []
    aucs = []
    title = "CCCS-CIC-AndMal-2020 Family Classification" if num_classes > 2 else "CCCS-CIC-AndMal-2020 Binary Classification"
    # 创建一个图形对象和子图对象
    fig, ax = plt.subplots()

    # 逐个计算每个类别的ROC曲线
    for i in range(num_classes):
        # 将当前类别视为正例，其余类别视为负例
        y_true_i = (y_true == i).astype(int)
        probas_i = y_pred[:, i]

        # 计算真正例率、假正例率和阈值
        fpr, tpr, thresholds = roc_curve(y_true_i, probas_i)

        # 将当前类别的TPR、FPR、AUC记录下来
        tprs.append(tpr)
        fprs.append(fpr)
        aucs.append(auc(fpr, tpr))

        # 绘制当前类别的ROC曲线
        ax.plot(fpr, tpr, label=f'Class {i} (AUC = {aucs[i]:.2f})')
    # 设置图例和标签
    ax.legend(loc='lower right')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Each Class')
    plt.savefig(f"./ROC/AndMal/{title}_{num}.svg", format='svg', dpi=1200)
    # 显示图形
    plt.show()


def draw_confusion_matrix(y_test, y_pred, num_classes, num):
    from matplotlib.colors import Normalize

    classes = ['Adware', 'Backdoor', 'FileInfector', 'No_Category', 'PUA',
               'Ransomware', 'Riskware', 'Scareware', 'Trojan',
               'Trojan_Banker', 'Trojan_Dropper', 'Trojan_SMS', 'Trojan_Spy', 'Zero_Day'] if num_classes > 2 else [
        'begine', 'malware']

    title = "CCCS-CIC-AndMal-2020 Family Classification" if num_classes > 2 else "CCCS-CIC-AndMal-2020 Binary Classification"
    # 计算混淆矩阵

    # 定义自定义的颜色映射范围

    cm = confusion_matrix(y_test, y_pred)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    # 绘制混淆矩阵图
    plt.figure(figsize=(14, 11))
    plt.imshow(cm, cmap=plt.cm.Reds)
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # 添加刻度标签
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60)
    plt.yticks(tick_marks, classes)

    # 添加文本标签（包括百分比）
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 获取当前分类的数量
        count = cm[i, j]
        # 计算百分比，并格式化为字符串
        percentage = "{:.2%}".format(count)
        # 设置文本标签
        plt.text(j, i, f"{percentage}",
                 horizontalalignment="center",
                 )

    # 保存图像，并设置dpi
    if not os.path.exists('./confusionPlt/CCCS-CIC-AndMal-2020/'):
        os.makedirs('./confusionPlt/CCCS-CIC-AndMal-2020/')
    plt.savefig(f"./confusionPlt/CCCS-CIC-AndMal-2020/{title}_{num}.svg", format='svg', dpi=1200)

    plt.show()


from memory_profiler import profile


@profile
def compute_classifier(lbl, X, y):
    start = timeit.default_timer()
    model = BoostForestClassifier(max_leafs=5, min_sample_leaf_list=5, reg_alpha_list=0.1, n_estimators=20)
    # model = BoostTreeClassifier(max_leafs=None, min_sample_leaf_list=5, reg_alpha_list=0.1)
    # model = RandomForestClassifier()
    # model = XGBClassifier()
    # model = LGBMClassifier()
    # model = DecisionTreeClassifier()
    # model = SVC()
    # model = KNeighborsClassifier(n_neighbors=6)

    # 创建XGBoost分类模型
    # model = XGBClassifier()

    # 设置十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # 初始化列表来存储每个折叠的性能指标
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    fpr_list = []

    # 创建空白图像对象
    fig, ax = plt.subplots()
    i = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 训练模型，并监测训练集和测试集上的性能
        # eval_set = [(X_train, y_train), (X_test, y_test)]
        # model.fit(X_train, y_train, eval_metric=["mlogloss"], eval_set=eval_set)

        model.fit(X_train, y_train)

        # 获取训练损失和验证损失的历史记录
        # history = model.evals_result()

        # 提取训练损失和验证损失
        # train_loss = history['validation_0']['mlogloss']
        # test_loss = history['validation_1']['mlogloss']

        # 添加子图并绘制损失下降曲线
        # ax.plot(train_loss, label='Train')
        # ax.plot(test_loss, label='Test')

        # 设置图例
        # ax.legend()
        # ax.set_xlabel('Number of iterations')
        # ax.set_ylabel('Log Loss')
        # ax.set_title('Training Loss')
        #
        # savePngPath = './XGBoost_Loss/'
        # if not os.path.exists(savePngPath):
        #     os.mkdir(savePngPath)
        # plt.savefig(savePngPath + f'XGboost_Loss_MalMem_{i}.png')
        # plt.show()

        # 预测
        y_pred = model.predict(X_test)

        counts = len(set(y))

        predict_proba = model.predict_proba(X_test)
        plot_roc_curves(y_test, predict_proba, counts, i)

        draw_confusion_matrix(y_test, y_pred, counts, i)
        i = i + 1

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

    print([mean_fpr, mean_recall, mean_precision, mean_accuracy, mean_f1])
    # compute_metrics(model_name, y_true, y_score)

    end = timeit.default_timer()
    print(f'used time : {end - start}s')

    return [mean_fpr, mean_recall, mean_precision, mean_accuracy, mean_f1]


def tenKFold(model, X, y):
    # 设置十折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    # 初始化列表来存储每个折叠的性能指标
    precision_list = []
    recall_list = []
    f1_list = []
    accuracy_list = []
    fpr_list = []
    time_train_list = []
    time_test_list = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        start = timeit.default_timer()
        model.fit(X_train, y_train)
        end = timeit.default_timer()
        time_train_list.append(end - start)

        start = timeit.default_timer()
        y_pred = model.predict(X_test)
        end = timeit.default_timer()
        time_test_list.append(end - start)

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
    mean_time_train = np.mean(time_train_list)
    mean_time_test = np.mean(time_test_list)
    return mean_precision, mean_recall, mean_f1, mean_accuracy, mean_fpr, mean_time_train, mean_time_test


def compute_cls(name, cls, X, y):
    RF = RandomForestClassifier()
    XGBC = XGBClassifier()
    LGBC = LGBMClassifier()
    DT = DecisionTreeClassifier()
    SVM = SVC()
    KNN = KNeighborsClassifier(n_neighbors=6)
    mean_accuracy = []
    mean_precision = []
    mean_recall = []
    mean_f1 = []
    mean_fpr = []
    mean_time_train = []
    mean_time_test = []
    for model in [RF, XGBC, LGBC, DT, SVM, KNN]:
        precision, recall, f1, accuracy, fpr, time_train, time_test = tenKFold(model, X, y)
        mean_accuracy.append(accuracy)
        mean_precision.append(precision)
        mean_recall.append(recall)
        mean_f1.append(f1)
        mean_fpr.append(fpr)
        mean_time_train.append(time_train)
        mean_time_test.append(time_test)
    # 创建数据框
    data = {'Model': ['RF', 'XGBC', 'LGBC', 'DT', 'SVM', 'KNN'],
            'mean_accuracy': mean_accuracy,
            'mean_precision': mean_precision,
            'mean_recall': mean_recall,
            'mean_f1': mean_f1,
            'mean_fpr': mean_fpr,
            'mean_time_train': mean_time_train,
            'mean_time_test': mean_time_test,
            }

    df = pd.DataFrame(data)
    # 保存为Excel文件
    df.to_excel(f'{name}_{cls}_metrics.xlsx', index=False)


if __name__ == '__main__':
    FILE_PATH = '../data/AndMal2020.csv'
    savePath = './demin_AndMal/2/'
    name = 'AndMal_SAAE'
    cls = '2'
    workdir = savePath
    dim = 8

    paths = ['X_train_ASSDAE_split_test.csv']
    paths = [workdir + str for str in paths]

    # 读取 CSV 文件，跳过第一行
    metriSaveList = [
        ['降维方法', 'mean_fpr', 'mean_recall', 'mean_precision', 'mean_accuracy', 'mean_f1', '时间开销']
    ]

    for index, path in enumerate(paths):
        X = pd.read_csv(path, header=None, skiprows=1, dtype=float).iloc[:, 0:dim - 1]
        y = pd.read_csv(path, header=None, skiprows=1).iloc[:, dim]
        compute_cls(name, cls, X, y)