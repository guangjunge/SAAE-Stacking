import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from memory_profiler import profile
@profile
def text():
    # 创建一个示例数据集
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建基础学习器
    nb_clf = GaussianNB()
    dt_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier()

    # 训练基础学习器
    nb_clf.fit(X_train, y_train)
    dt_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    # 得到基础学习器的预测结果
    nb_preds = nb_clf.predict(X_test)
    dt_preds = dt_clf.predict(X_test)
    rf_preds = rf_clf.predict(X_test)

    # 创建元学习器的训练集
    meta_X_train = np.column_stack((nb_preds, dt_preds, rf_preds))

    # 创建元学习器（逻辑回归）并训练
    meta_clf = LogisticRegression()
    meta_clf.fit(meta_X_train, y_test)

    # 得到基础学习器的预测结果作为元学习器的输入
    meta_X_test = np.column_stack((nb_clf.predict(X_test), dt_clf.predict(X_test), rf_clf.predict(X_test)))

    # 使用元学习器进行最终预测
    meta_preds = meta_clf.predict(meta_X_test)

    # 计算元集成学习模型的准确率
    accuracy = accuracy_score(y_test, meta_preds)
    print("Meta-ensemble Accuracy:", accuracy)


def text1():
    # 创建一个示例数据集
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建基础学习器
    nb_clf = GaussianNB()
    dt_clf = DecisionTreeClassifier()
    rf_clf = RandomForestClassifier()

    # 训练基础学习器
    nb_clf.fit(X_train, y_train)
    dt_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    # 得到基础学习器的预测结果
    nb_preds = nb_clf.predict(X_test)
    dt_preds = dt_clf.predict(X_test)
    rf_preds = rf_clf.predict(X_test)

    # 创建元学习器的训练集
    meta_X_train = np.column_stack((nb_preds, dt_preds, rf_preds))

    # 创建元学习器（逻辑回归）并训练
    meta_clf = LogisticRegression()
    meta_clf.fit(meta_X_train, y_test)

    # 得到基础学习器的预测结果作为元学习器的输入
    meta_X_test = np.column_stack((nb_clf.predict(X_test), dt_clf.predict(X_test), rf_clf.predict(X_test)))

    # 使用元学习器进行最终预测
    meta_preds = meta_clf.predict(meta_X_test)

    # 计算元集成学习模型的准确率
    accuracy = accuracy_score(y_test, meta_preds)
    print("Meta-ensemble Accuracy:", accuracy)





if __name__ == "__main__":
    text()

