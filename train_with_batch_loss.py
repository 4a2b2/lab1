# -*- coding: utf-8 -*-
"""
手写 mini-batch 梯度下降训练 逻辑回归 / 线性SVM / MLP，
逐批记录真实训练损失（每批样本不同，损失自然振荡），
并绘制三种模型的逐批损失曲线图。

说明：朴素贝叶斯是基于计数统计的闭式解，无迭代训练过程，
      因此不存在训练损失曲线。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BATCH = 128


def load_features():
    """加载并预处理数据，返回稀疏特征与标签"""
    import main  # 复用 main.py 的数据加载与预处理
    X_raw, y, _ = main.load_data()
    X_clean = main.preprocess_texts(X_raw)
    X_tr_text, X_val_text, y_train, y_val = train_test_split(
        X_clean, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words='english')
    X_train = vec.fit_transform(X_tr_text)   # 稀疏矩阵, L2归一化
    X_val = vec.transform(X_val_text)
    return X_train, X_val, y_train, y_val


def make_iterator(X_train, Y_oh, N):
    def iterate_batches():
        idx = rng.permutation(N)
        for s in range(0, N, BATCH):
            b = idx[s:s + BATCH]
            yield b, X_train[b], Y_oh[b]
    return iterate_batches


def moving_average(x, w=50):
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().values


# ======================================================================
# 1. 逻辑回归（softmax + 交叉熵），mini-batch GD
# ======================================================================
def train_logistic(iterate_batches, X_val, y_val, D, K,
                   lr=1.0, reg=1e-4, epochs=80, verbose=False):
    W = np.zeros((D, K))
    b = np.zeros(K)
    losses = []
    for ep in range(epochs):
        for bidx, Xb, yb in iterate_batches():
            m = len(bidx)
            logits = (Xb @ W) + b
            logits = logits - logits.max(axis=1, keepdims=True)   # 数值稳定
            exp = np.exp(logits)
            p = exp / exp.sum(axis=1, keepdims=True)
            loss = -np.mean(np.sum(yb * np.log(p + 1e-12), axis=1)) + 0.5 * reg * np.sum(W * W)
            losses.append(loss)
            g = (p - yb) / m
            W -= lr * (Xb.T @ g + reg * W)
            b -= lr * g.sum(axis=0)
        acc = accuracy_score(y_val, np.argmax((X_val @ W) + b, axis=1))
        if verbose and (ep + 1) % 10 == 0:
            print(f'  [LR] epoch {ep+1:3d}  loss={losses[-1]:.4f}  val_acc={acc:.4f}')
    return losses, acc


# ======================================================================
# 2. 线性 SVM（OvR hinge loss + L2），mini-batch 次梯度下降
# ======================================================================
def train_svm(iterate_batches, X_val, y_val, D, K,
              lr=0.2, reg=1e-4, epochs=60, verbose=False):
    W = np.zeros((D, K))
    b = np.zeros(K)
    losses = []
    for ep in range(epochs):
        for bidx, Xb, yb_oh in iterate_batches():
            m = len(bidx)
            yb = np.where(yb_oh > 0, 1.0, -1.0)          # ±1
            scores = (Xb @ W) + b                        # (m,K)
            margin = yb * scores
            viol = np.maximum(0.0, 1.0 - margin)         # hinge
            loss = 0.5 * reg * np.sum(W * W) + np.mean(np.sum(viol, axis=1))
            losses.append(loss)
            mask = (viol > 0).astype(float)
            gscores = -yb * mask / m                     # d loss/d scores
            W -= lr * (Xb.T @ gscores + reg * W)
            b -= lr * gscores.sum(axis=0)
        acc = accuracy_score(y_val, np.argmax((X_val @ W) + b, axis=1))
        if verbose and (ep + 1) % 10 == 0:
            print(f'  [SVM] epoch {ep+1:3d}  loss={losses[-1]:.4f}  val_acc={acc:.4f}')
    return losses, acc


# ======================================================================
# 3. MLP（1隐层, ReLU, softmax CE），手写反向传播
# ======================================================================
def train_mlp(iterate_batches, X_val, y_val, D, K,
              hidden=100, lr=0.1, reg=1e-4, epochs=80, verbose=False):
    W1 = rng.normal(0, np.sqrt(2.0 / D), (D, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, K))
    b2 = np.zeros(K)
    losses = []
    for ep in range(epochs):
        for bidx, Xb, yb in iterate_batches():
            m = len(bidx)
            z1 = (Xb @ W1) + b1
            a1 = np.maximum(0, z1)                       # ReLU
            logits = a1 @ W2 + b2
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            p = exp / exp.sum(axis=1, keepdims=True)
            loss = -np.mean(np.sum(yb * np.log(p + 1e-12), axis=1)) \
                   + 0.5 * reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
            losses.append(loss)
            # 反向传播
            dlogits = (p - yb) / m
            dW2 = a1.T @ dlogits + reg * W2
            db2 = dlogits.sum(axis=0)
            da1 = dlogits @ W2.T
            dz1 = da1 * (z1 > 0)
            dW1 = Xb.T @ dz1 + reg * W1
            db1 = dz1.sum(axis=0)
            W2 -= lr * dW2; b2 -= lr * db2
            W1 -= lr * dW1; b1 -= lr * db1
        acc = accuracy_score(y_val,
                             np.argmax(np.maximum(0, (X_val @ W1) + b1) @ W2 + b2, axis=1))
        if verbose and (ep + 1) % 10 == 0:
            print(f'  [MLP] epoch {ep+1:3d}  loss={losses[-1]:.4f}  val_acc={acc:.4f}')
    return losses, acc


def main():
    print('=== 手写 mini-batch 训练（逐批记录损失）===')
    X_train, X_val, y_train, y_val = load_features()
    N, D = X_train.shape
    K = 10
    Y_oh = np.eye(K)[y_train]
    iterate_batches = make_iterator(X_train, Y_oh, N)
    print(f'训练集 {N} 条, 特征 {D} 维, batch={BATCH}, 每epoch约 {N // BATCH + 1} 步\n')

    print('--- 逻辑回归 ---')
    loss_lr, acc_lr = train_logistic(iterate_batches, X_val, y_val, D, K, verbose=True)
    print('--- 线性SVM ---')
    loss_svm, acc_svm = train_svm(iterate_batches, X_val, y_val, D, K, verbose=True)
    print('--- MLP ---')
    loss_mlp, acc_mlp = train_mlp(iterate_batches, X_val, y_val, D, K, verbose=True)

    print(f'\n最终验证集准确率: LR={acc_lr:.4f}  SVM={acc_svm:.4f}  MLP={acc_mlp:.4f}')

    # ---------- 绘图：1×3 子图，逐批损失（振荡）+ 移动平均趋势 ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    panels = [
        (loss_lr,  acc_lr,  f'逻辑回归 (val_acc={acc_lr:.4f})'),
        (loss_svm, acc_svm, f'线性SVM (val_acc={acc_svm:.4f})'),
        (loss_mlp, acc_mlp, f'MLP(100) (val_acc={acc_mlp:.4f})'),
    ]
    for i, (ax, (losses, _, title)) in enumerate(zip(axes, panels)):
        steps = np.arange(1, len(losses) + 1)
        ax.plot(steps, losses, color='#4C9BE2', lw=0.5, alpha=0.45, label='逐批损失')
        ax.plot(steps, moving_average(losses), color='#E2574C', lw=2.0, label='移动平均')
        if i == 1:  # SVM 初始 hinge 损失很高，截断 y 轴以看清振荡细节
            ax.set_ylim(0, 4)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('训练步数 (batch step)')
        ax.set_ylabel('损失')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle('三种模型 mini-batch 逐批训练损失曲线（振荡中下降）', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FIG_DIR, 'batch_loss_curves.png')
    fig.savefig(out, dpi=150)
    print(f'图已保存: {out}')


if __name__ == '__main__':
    main()
