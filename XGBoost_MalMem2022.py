import itertools
import os
import timeit
import warnings

# 忽略特定类型的警告
from openpyxl import Workbook
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")

from keras.layers import Input, LayerNormalization, Dropout, GaussianNoise
from lightgbm import LGBMClassifier
from sklearn import manifold, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import roc_curve, auc
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
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import numpy as np
import random
import tensorflow as tf
import pickle
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from keras.callbacks import TensorBoard
from sklearn import metrics
from sklearn.model_selection import cross_val_score, KFold
from ASSDAE import ASSDAE




def draw_confusion_matrix(y_test, y_pred, num_classes, num):

    classes = ['180solutions', 'Ako', 'CWS', 'Conti', 'Emotet', 'Gator', 'Maze', 'Pysa', 'Reconyc', 'Refroso', 'Scar',
              'Shade', 'TIBS', 'Transponder', 'Zeus'] if num_classes > 2 else ['begine', 'malware']

    title = "CIC-MalMen2022 Family Classification" if num_classes > 2 else "CIC-MalMen2022 Binary Classification"

    # 计算混淆矩阵
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
    if not os.path.exists('./confusionPlt/MalMem2022/'):
        os.makedirs('./confusionPlt/MalMem2022/')
    plt.savefig(f"./confusionPlt/MalMem2022/'{title}_{num}.svg", format='svg', dpi=1200)
    plt.show()

def plot_roc_curves(y_true, y_pred, num_classes, num):
    # 初始化真正例率（TPR）、假正例率（FPR）和AUC
    tprs = []
    fprs = []
    aucs = []
    title = "CIC-MalMem-2022 Family Classification" if num_classes > 2 else "CIC-MalMem-2022 Binary Classification"
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
    plt.savefig(f"./ROC/MalMem/{title}_{num}.svg", format='svg', dpi=1200)
    # 显示图形
    plt.show()


from memory_profiler import profile

@profile
def compute_classifier(lbl, X, y):
    start = timeit.default_timer()

    # model = RandomForestClassifier()
    model = XGBClassifier()
    # model = LGBMClassifier()
    # model = DecisionTreeClassifier()
    # model = SVC()
    # model = KNeighborsClassifier(n_neighbors=6)

    # # 创建XGBoost分类模型
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

        # 使用numpy的unique函数计算不同元素个数
        counts = len(set(y))
        print(counts)

        predict_proba = model.predict_proba(X_test)
        plot_roc_curves(y_test, predict_proba, counts,i)

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


    # 设置图例
    ax.legend()
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Log Loss')
    ax.set_title('Training Loss')

    savePngPath = './XGBoost_Loss/'
    if not os.path.exists(savePngPath):
        os.mkdir(savePngPath)
    plt.savefig(savePngPath + 'XGboost_Loss_MalMem.png')
    plt.show()

    print(lbl)
    print([mean_fpr, mean_recall, mean_precision, mean_accuracy, mean_f1])
    end = timeit.default_timer()
    print(f'used time : {end - start}s')
    return [mean_fpr, mean_recall, mean_precision, mean_accuracy, mean_f1]

import xgboost as xgb
import lightgbm as lgb
def stackLearn(X, y):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义基分类器和元分类器
    xgb_clf = XGBClassifier()
    lgb_clf = LGBMClassifier()
    rf_clf = RandomForestClassifier()
    meta_clf = svm.SVC(kernel='linear', C=1)

    # model = XGBClassifier()
    # model = LGBMClassifier()

    # 创建空数组，用于存储每个基分类器的预测结果
    train_meta_features = np.zeros((len(X_train), 2))

    # 分别在XGBoost和LightGBM上训练基分类器
    xgb_clf.fit(X_train, y_train)
    lgb_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    # 在训练集的测试部分上进行预测，并存储预测结果
    xgb_pred = xgb_clf.predict_proba(X_test)[:, 1]
    lgb_pred = lgb_clf.predict_proba(X_test)[:, 1]
    rf_pred = rf_clf.predict_proba(X_test)[:, 1]


    # xgb_pred = xgb_clf.predict(X_test)
    # lgb_pred = lgb_clf.predict(X_test)
    # rf_pred  = rf_clf.predict(X_test)
    #
    # xgbacc = accuracy_score(y_test, xgb_pred)
    # lgbacc = accuracy_score(y_test, lgb_pred)
    # rfacc = accuracy_score(y_test, rf_pred)
    # print("xgbacc: {:.3f}".format(xgbacc))
    # print("lgbacc: {:.3f}".format(lgbacc))
    # print("rfacc: {:.3f}".format(rfacc))

    # print(xgb_clf.predict_proba(X_test))
    #
    # # 将两个基分类器的预测结果按列组合成一个特征矩阵
    meta_features = np.column_stack((xgb_pred, lgb_pred, rf_pred))

    # 使用元分类器对组合后的特征矩阵进行训练
    meta_clf.fit(meta_features, y_test)

    # 在测试集上进行预测
    test_meta_features = np.column_stack((xgb_clf.predict_proba(X_test)[:, 1], lgb_clf.predict_proba(X_test)[:, 1], rf_clf.predict_proba(X_test)[:, 1]))
    meta_pred = meta_clf.predict(test_meta_features)

    acc = accuracy_score(y_test, meta_pred)
    report = classification_report(y_test, meta_pred)

    print("Accuracy: {:.3f}".format(acc))
    print("Classification Report:")
    print(report)



if __name__ == '__main__':
    FILE_PATH = '../data/CIC-MalMem2022.csv'
    # savePath = './demin_MalMem/2/'
    savePath = './demin_AndMal/2/'
    workdir = savePath

    dim = 4

    # paths = ['X_train_PCA.csv', 'X_train_UMAP.csv', 'X_train_ASSDAE.csv']

    paths = ['X_train_ASSDAE_split_test.csv']
    paths = [workdir+str for str in paths]

    # 读取 CSV 文件，跳过第一行
    metriSaveList = [
        ['降维方法', 'mean_fpr', 'mean_recall' , 'mean_precision', 'mean_accuracy', 'mean_f1', '时间开销']
    ]

    for index, path in enumerate(paths):
        X = pd.read_csv(path, header=None, skiprows=1, dtype=float).iloc[:, 0:dim-1]
        y = pd.read_csv(path, header=None, skiprows=1).iloc[:, dim]

        stackLearn(X, y)

    #     metri = compute_classifier(path, X, y)
    #     method = [path.split('train_')[1].split('.')[0]]
    #     metriSaveList.append(method + metri)
    #
    # # 创建一个新的工作簿
    # workbook = Workbook()
    # # 获取默认的活动工作表（第一个工作表）
    # sheet = workbook.active
    # # 将列表数据写入表格
    # for row in metriSaveList:
    #     sheet.append(row)
    # # 保存工作簿到指定文件
    # workbook.save('Metri_MalMem.xlsx')
















    # y = pd.read_csv(paths[0], header=None, skiprows=1).iloc[:, dim]
    # X_train_PCA = pd.read_csv(paths[0], header=None, skiprows=1, dtype=float).iloc[:, 0:dim-1]
    # X_train_UMAP = pd.read_csv(paths[1], header=None, skiprows=1, dtype=float).iloc[:, 0:dim-1]
    # # X_train_TSNE = pd.read_csv(paths[2], header=None, skiprows=1, dtype={0: float, 1: float, 2: int}).iloc[:, 0:2]
    # X_train_ASSDAE = pd.read_csv(paths[2], header=None, skiprows=1, dtype=float).iloc[:, 0:dim-1]
    #
    # print('draw plot')
    # classs = ['Ransomware', 'Spyware', 'Trojan Horse ']
    # Begin   = ['Begin', 'Malware']
    # family = ['180solutions', 'Ako', 'CWS', 'Conti', 'Emotet',
    #            'Gator','Maze','Pysa','Reconyc','Refroso',
    #            'Scar','Shade','TIBS','Transponder','Zeus',]
    #
    #
    # fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    # axes[0].title.set_text('ASSAE')
    # axes[0].scatter(X_train_ASSDAE.iloc[:, 0], X_train_ASSDAE.iloc[:, 1], c=y, s=100, label=y)
    #
    # axes[1].title.set_text('PCA')
    # axes[1].scatter(X_train_PCA.iloc[:, 0], X_train_PCA.iloc[:, 1], c=y, s=100, label=y)
    #
    # # axes[2].title.set_text('TSNE')
    # # axes[2].scatter(X_train_TSNE.iloc[:, 0], X_train_TSNE.iloc[:, 1], c=y, s=100, label=y)
    #
    # axes[2].title.set_text('UMAP')
    # scatter = axes[2].scatter(X_train_UMAP.iloc[:, 0], X_train_UMAP.iloc[:, 1], c=y, s=100, label=y)
    #
    # # 添加图例
    # handles, labels = scatter.legend_elements()
    # axes[0].legend(handles, classs)
    # axes[1].legend(handles, classs)
    # axes[2].legend(handles, classs)
    #
    # # 显示图像
    plt.show()



