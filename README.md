# 实验一：文本分类 — 运行说明

## 环境要求

- Python 3.14（实验实际使用 3.14.8）
- 依赖库：numpy 2.5、pandas 3.0、scikit-learn 1.9、matplotlib、seaborn

安装命令（使用 conda）：
```bash
conda create -n project1 python=3.12
conda activate project1
pip install numpy pandas scikit-learn matplotlib seaborn
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 主实验脚本，包含完整实验流程 |
| `train_with_batch_loss.py` | 手写 mini-batch 梯度下降训练 LR/SVM/MLP，逐批记录损失并绘图 |
| `train_data.csv` | 训练集（7368条，10类） |
| `test_data_unlabeled.csv` | 无标签测试集（2457条） |
| `predictions.csv` | 测试集预测结果（最优模型生成，2457行） |
| `TUNING.md` | 超参数调参指南（课程提供） |
| `figures/` | 可视化图表输出目录（6张图） |
| `实验报告.docx` / `实验报告.pdf` | 实验报告（2页） |

## 运行方式

```bash
cd data_project1
python main.py                 # 主实验（约20分钟）
python train_with_batch_loss.py  # 手写mini-batch损失曲线（约5分钟）
```

`main.py` 依次执行：
1. 数据加载与探索
2. 文本预处理（去邮件头、提取Subject、清洗符号）
3. 训练/验证集划分（80/20 分层）
4. TF-IDF 特征提取（max_features=20000, ngram_range=(1,2)）
5. 四种模型基线实验
6. TF-IDF 参数消融对比
7. SVM/LR/MLP 超参数调优
8. 最优模型选择与测试集预测

## 主要结果（scikit-learn 1.9.1）

| 模型 | 最优超参数 | 验证集准确率 |
|------|-----------|-------------|
| 朴素贝叶斯 | 默认 | 0.8562 |
| SVM (linear) | C=1 | 0.8602 |
| 逻辑回归 | C=10 | 0.8670 |
| **MLP** | **隐藏层 (200,)** | **0.8704** |

手写 mini-batch 训练（batch=128）验证集准确率：LR 84.26% / SVM 83.04% / MLP 84.33%，与 sklearn 实现相当。

## 输出文件

- `predictions.csv`：测试集预测结果（每行一个类别标签）
- `figures/class_distribution.png`：训练集类别分布
- `figures/baseline_comparison.png`：基线模型性能对比
- `figures/svm_tuning.png`：SVM 参数 C 调优曲线
- `figures/lr_tuning.png`：LR 参数 C 调优曲线
- `figures/confusion_matrix.png`：最优模型验证集混淆矩阵
- `figures/batch_loss_curves.png`：三种模型逐批训练损失曲线（振荡中下降）

## 复现说明

- 随机种子固定为 42（np.random.seed 和各模型 random_state）
- 数据划分使用 stratify 分层抽样，保证类别比例一致
- TF-IDF 仅在训练子集上 fit，验证集和测试集用同一规则 transform
