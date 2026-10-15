# -*- coding: utf-8 -*-
"""
当代人工智能实验课 - 实验一：文本分类
=====================================
本脚本完成新闻文本10分类任务，包含：
  1. 数据加载、划分与探索
  2. 文本预处理（去除邮件头、清洗文本）
  3. TF-IDF 特征工程
  4. 四种分类模型训练（朴素贝叶斯、SVM、逻辑回归、MLP）
  5. 超参数调优与模型比较（基于验证集）
  6. 最优模型在测试集上预测并保存结果
  7. 可视化生成

固定随机种子，保证实验可复现。
运行方式: python main.py
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 无界面环境下保存图片
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

# 固定随机种子，保证可复现
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# 路径设置（基于本文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, 'train_data.csv')
TEST_CSV = os.path.join(BASE_DIR, 'test_data_unlabeled.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'predictions.csv')
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ======================================================================
# 第一部分：数据加载与探索
# ======================================================================
def load_data():
    """读取训练集和无标签测试集"""
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    X_train = train_df['text'].astype(str).tolist()
    y_train = train_df['target'].values
    X_test = test_df['text'].astype(str).tolist()
    return X_train, y_train, X_test


# ======================================================================
# 第二部分：文本预处理
# ======================================================================
def clean_text(text):
    """
    清洗单条文本：
      - 保留 Subject 行内容（有区分力）
      - 去除邮件头信息（From/Organization/Lines 等字段）
      - 去除引用行（以 > 开头）
      - 去除邮箱地址、特殊符号
      - 转小写
    """
    # 提取 Subject 行内容
    subject_match = re.search(r'^Subject:\s*(.*?)(?=\n\S)', text, re.DOTALL | re.IGNORECASE)
    subject = subject_match.group(1).strip() if subject_match else ''

    # 去除所有邮件头字段（直到第一个空行）
    parts = re.split(r'\n\n', text, maxsplit=1)
    body = parts[1] if len(parts) > 1 else text

    # 去除引用行
    body = re.sub(r'(?m)^>.*$', '', body)
    # 去除邮箱
    body = re.sub(r'\S+@\S+', ' ', body)
    # 去除非字母字符（保留空格）
    body = re.sub(r'[^a-zA-Z\s]', ' ', body)

    combined = (subject + ' ' + body).lower()
    # 合并多余空格
    combined = re.sub(r'\s+', ' ', combined).strip()
    return combined


def preprocess_texts(texts):
    """批量预处理"""
    return [clean_text(t) for t in texts]


# ======================================================================
# 第三部分：模型训练与评估工具
# ======================================================================
def evaluate_model(model, X_train, y_train, X_val, y_val, name=''):
    """训练并在验证集上评估，返回准确率和加权F1"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    return acc, f1, y_pred


def print_result(name, acc, f1):
    print(f"  {name:<35s} | 验证集准确率: {acc:.4f} | 加权F1: {f1:.4f}")


# ======================================================================
# 第四部分：基线实验（默认参数）
# ======================================================================
def run_baselines(X_train_tfidf, y_train, X_val_tfidf, y_val):
    """使用各模型默认参数建立基线"""
    print("\n" + "=" * 60)
    print("第一阶段：基线实验（默认参数）")
    print("=" * 60)

    baselines = {
        'NaiveBayes': MultinomialNB(),
        'SVM(linear,C=1)': SVC(kernel='linear', C=1.0, random_state=RANDOM_STATE),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'MLP(100,)': MLPClassifier(hidden_layer_sizes=(100,), max_iter=300,
                                   random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in baselines.items():
        acc, f1, _ = evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val)
        print_result(name, acc, f1)
        results[name] = {'acc': acc, 'f1': f1}
    return results


# ======================================================================
# 第五部分：超参数调优
# ======================================================================
def tune_models(X_train_tfidf, y_train, X_val_tfidf, y_val):
    """对每个模型在验证集上搜索超参数"""
    print("\n" + "=" * 60)
    print("第二阶段：超参数调优（在验证集上比较）")
    print("=" * 60)

    # --- SVM 调优: C ---
    print("\n--- SVM 调优 (kernel=linear) ---")
    svm_results = {}
    for C in [0.1, 1, 10, 100]:
        model = SVC(kernel='linear', C=C, random_state=RANDOM_STATE)
        acc, f1, _ = evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val)
        key = f'SVM(C={C})'
        print_result(key, acc, f1)
        svm_results[key] = {'acc': acc, 'f1': f1, 'C': C}

    # --- 逻辑回归调优: C ---
    print("\n--- LogisticRegression 调优 ---")
    lr_results = {}
    for C in [0.1, 1, 10, 100]:
        model = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)
        acc, f1, _ = evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val)
        key = f'LR(C={C})'
        print_result(key, acc, f1)
        lr_results[key] = {'acc': acc, 'f1': f1, 'C': C}

    # --- MLP 调优: 隐藏层结构 ---
    print("\n--- MLP 调优 ---")
    mlp_results = {}
    for hidden in [(50,), (100,), (200,), (100, 50)]:
        model = MLPClassifier(hidden_layer_sizes=hidden, max_iter=300,
                              random_state=RANDOM_STATE)
        acc, f1, _ = evaluate_model(model, X_train_tfidf, y_train, X_val_tfidf, y_val)
        key = f'MLP{hidden}'
        print_result(key, acc, f1)
        mlp_results[key] = {'acc': acc, 'f1': f1, 'hidden': hidden}

    return svm_results, lr_results, mlp_results


# ======================================================================
# 第六部分：TF-IDF 参数对比（消融）
# ======================================================================
def compare_tfidf_params(X_train_clean, y_train, X_val_clean, y_val):
    """对比不同 TF-IDF 配置对性能的影响"""
    print("\n" + "=" * 60)
    print("第三阶段：TF-IDF 参数对比（以 SVM linear C=10 为基准）")
    print("=" * 60)

    configs = {
        'TF-IDF(5000,uni)': {'max_features': 5000, 'ngram_range': (1, 1)},
        'TF-IDF(10000,uni)': {'max_features': 10000, 'ngram_range': (1, 1)},
        'TF-IDF(20000,uni+bi)': {'max_features': 20000, 'ngram_range': (1, 2)},
    }

    results = {}
    for name, cfg in configs.items():
        vec = TfidfVectorizer(max_features=cfg['max_features'],
                              ngram_range=cfg['ngram_range'],
                              stop_words='english')
        Xt = vec.fit_transform(X_train_clean)
        Xv = vec.transform(X_val_clean)
        model = SVC(kernel='linear', C=10, random_state=RANDOM_STATE)
        acc, f1, _ = evaluate_model(model, Xt, y_train, Xv, y_val)
        print_result(name, acc, f1)
        results[name] = {'acc': acc, 'f1': f1, 'cfg': cfg}
    return results


# ======================================================================
# 第七部分：可视化
# ======================================================================
def plot_baseline_comparison(baseline_results, save_path):
    """绘制基线模型对比柱状图"""
    names = list(baseline_results.keys())
    accs = [baseline_results[n]['acc'] for n in names]
    f1s = [baseline_results[n]['f1'] for n in names]

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, accs, width, label='准确率', color='#4C9BE2')
    bars2 = ax.bar(x + width / 2, f1s, width, label='加权F1', color='#F0A23A')
    ax.set_ylabel('分数')
    ax.set_title('基线模型验证集性能对比')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.0)
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_tuning_curve(results_dict, param_name, param_values, title, save_path):
    """绘制超参数调优曲线"""
    accs = [results_dict[f'{param_name}(C={v})']['acc'] for v in param_values]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(param_values)), accs, 'o-', color='#4C9BE2', linewidth=2, markersize=8)
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([str(v) for v in param_values])
    ax.set_xlabel('参数值')
    ax.set_ylabel('验证集准确率')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    for i, acc in enumerate(accs):
        ax.annotate(f'{acc:.4f}', (i, acc), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=range(10), yticklabels=range(10))
    ax.set_xlabel('预测类别')
    ax.set_ylabel('真实类别')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_class_distribution(y, title, save_path):
    """绘制类别分布"""
    unique, counts = np.unique(y, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(unique, counts, color='#4C9BE2')
    ax.set_xlabel('类别')
    ax.set_ylabel('样本数')
    ax.set_title(title)
    ax.set_xticks(unique)
    for bar in bars:
        ax.annotate(f'{bar.get_height()}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ======================================================================
# 主流程
# ======================================================================
def main():
    # ------ 1. 数据加载 ------
    print("=" * 60)
    print("第一步：数据加载")
    print("=" * 60)
    X_raw, y, X_test_raw = load_data()
    print(f"训练集样本数: {len(X_raw)}")
    print(f"测试集样本数: {len(X_test_raw)}")
    print(f"类别数: {len(np.unique(y))}")

    # 类别分布可视化
    plot_class_distribution(y, '训练集类别分布', os.path.join(FIG_DIR, 'class_distribution.png'))

    # ------ 2. 文本预处理 ------
    print("\n" + "=" * 60)
    print("第二步：文本预处理")
    print("=" * 60)
    X_clean = preprocess_texts(X_raw)
    X_test_clean = preprocess_texts(X_test_raw)
    print("预处理完成（去除邮件头、引用行、邮箱、符号；提取Subject）")
    print(f"清洗前样本(前120字): {X_raw[0][:120]}")
    print(f"清洗后样本(前120字): {X_clean[0][:120]}")

    # ------ 3. 划分训练集与验证集（分层抽样 80/20） ------
    print("\n" + "=" * 60)
    print("第三步：划分训练集与验证集 (80/20, 分层)")
    print("=" * 60)
    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_clean, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"训练集: {len(X_train_text)} | 验证集: {len(X_val_text)}")

    # ------ 4. TF-IDF 特征提取（仅在训练集上fit） ------
    print("\n" + "=" * 60)
    print("第四步：TF-IDF 特征工程")
    print("=" * 60)
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                 stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_val_tfidf = vectorizer.transform(X_val_text)
    X_test_tfidf = vectorizer.transform(X_test_clean)
    print(f"特征维度: {X_train_tfidf.shape[1]}")
    print(f"词表大小: {len(vectorizer.vocabulary_)}")

    # ------ 5. 基线实验 ------
    baseline_results = run_baselines(X_train_tfidf, y_train, X_val_tfidf, y_val)
    plot_baseline_comparison(baseline_results, os.path.join(FIG_DIR, 'baseline_comparison.png'))

    # ------ 6. TF-IDF 参数对比 ------
    tfidf_results = compare_tfidf_params(X_train_text, y_train, X_val_text, y_val)

    # ------ 7. 超参数调优 ------
    svm_results, lr_results, mlp_results = tune_models(
        X_train_tfidf, y_train, X_val_tfidf, y_val)

    # SVM 调优曲线
    plot_tuning_curve(svm_results, 'SVM', [0.1, 1, 10, 100],
                      'SVM 正则化参数 C 调优曲线',
                      os.path.join(FIG_DIR, 'svm_tuning.png'))

    # LR 调优曲线
    plot_tuning_curve(lr_results, 'LR', [0.1, 1, 10, 100],
                      'LogisticRegression 正则化参数 C 调优曲线',
                      os.path.join(FIG_DIR, 'lr_tuning.png'))

    # ------ 8. 汇总所有结果，选择最优 ------
    print("\n" + "=" * 60)
    print("第四阶段：模型选择")
    print("=" * 60)

    all_results = {}
    all_results.update(baseline_results)
    for name, r in svm_results.items():
        all_results[name] = r
    for name, r in lr_results.items():
        all_results[name] = r
    for name, r in mlp_results.items():
        all_results[name] = r

    best_name = max(all_results, key=lambda k: all_results[k]['acc'])
    best_acc = all_results[best_name]['acc']
    print(f"\n最优模型: {best_name} (验证集准确率={best_acc:.4f})")

    # ------ 9. 用最优配置在全量训练集上重新训练，预测测试集 ------
    print("\n" + "=" * 60)
    print("第五阶段：最终模型训练与测试集预测")
    print("=" * 60)
    print("使用最优超参数在全量训练集(训练+验证)上重新训练...")

    # 在全量训练集上重新 fit TF-IDF（避免数据泄露：测试集用同一规则transform）
    final_vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                                       stop_words='english')
    X_full_tfidf = final_vectorizer.fit_transform(X_clean)
    X_test_final = final_vectorizer.transform(X_test_clean)

    # 根据最优名称确定最终模型
    if 'SVM' in best_name:
        best_C = svm_results[best_name]['C']
        final_model = SVC(kernel='linear', C=best_C, random_state=RANDOM_STATE)
    elif best_name.startswith('LR'):
        best_C = lr_results[best_name]['C']
        final_model = LogisticRegression(C=best_C, max_iter=2000,
                                         random_state=RANDOM_STATE)
    elif 'MLP' in best_name:
        best_hidden = mlp_results[best_name]['hidden']
        final_model = MLPClassifier(hidden_layer_sizes=best_hidden, max_iter=300,
                                    random_state=RANDOM_STATE)
    else:
        final_model = MultinomialNB()

    final_model.fit(X_full_tfidf, y)
    predictions = final_model.predict(X_test_final)

    # 保存预测结果
    pd.DataFrame(predictions).to_csv(OUTPUT_CSV, index=False, header=False)
    print(f"预测结果已保存至: {OUTPUT_CSV}")
    print(f"预测类别分布:\n{pd.Series(predictions).value_counts().sort_index()}")

    # ------ 10. 验证集混淆矩阵与分类报告（用最优模型重新训练于训练子集） ------
    print("\n" + "=" * 60)
    print("验证集详细评估（最优模型）")
    print("=" * 60)
    if 'SVM' in best_name:
        eval_model = SVC(kernel='linear', C=svm_results[best_name]['C'],
                         random_state=RANDOM_STATE)
    elif best_name.startswith('LR'):
        eval_model = LogisticRegression(C=lr_results[best_name]['C'], max_iter=2000,
                                        random_state=RANDOM_STATE)
    elif 'MLP' in best_name:
        eval_model = MLPClassifier(hidden_layer_sizes=mlp_results[best_name]['hidden'],
                                   max_iter=300, random_state=RANDOM_STATE)
    else:
        eval_model = MultinomialNB()

    eval_model.fit(X_train_tfidf, y_train)
    y_val_pred = eval_model.predict(X_val_tfidf)
    print("\n分类报告:")
    print(classification_report(y_val, y_val_pred, digits=4))

    plot_confusion_matrix(y_val, y_val_pred,
                          f'最优模型({best_name})验证集混淆矩阵',
                          os.path.join(FIG_DIR, 'confusion_matrix.png'))

    print("\n" + "=" * 60)
    print("实验完成！所有图表保存在 figures/ 目录")
    print("=" * 60)


if __name__ == '__main__':
    main()
