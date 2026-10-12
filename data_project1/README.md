# 实验一：文本分类 — 运行说明

## 环境要求

- Python >= 3.7（推荐 3.10）
- 依赖库：numpy, pandas, scikit-learn, matplotlib, seaborn

安装命令（使用 conda）：
```bash
conda create -n project1 python=3.10
conda activate project1
conda install numpy pandas scikit-learn matplotlib seaborn
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 主实验脚本，包含完整实验流程 |
| `train_data.csv` | 训练集（7368条，10类） |
| `test_data_unlabeled.csv` | 无标签测试集（2457条） |
| `predictions.csv` | 测试集预测结果（最优模型生成） |
| `TUNING.md` | 超参数调参指南 |
| `figures/` | 可视化图表输出目录 |
| `实验报告.html` | 实验报告（浏览器打开查看） |

## 运行方式

```bash
cd data_project1
python main.py
```

运行后脚本将依次执行：
1. 数据加载与探索
2. 文本预处理（去邮件头、提取Subject、清洗符号）
3. 训练/验证集划分（80/20 分层）
4. TF-IDF 特征提取（max_features=20000, ngram_range=(1,2)）
5. 四种模型基线实验
6. TF-IDF 参数对比
7. SVM/LR/MLP 超参数调优
8. 最优模型选择与测试集预测

## 输出文件

- `predictions.csv`：测试集预测结果（每行一个类别标签）
- `figures/class_distribution.png`：训练集类别分布
- `figures/baseline_comparison.png`：基线模型性能对比
- `figures/svm_tuning.png`：SVM 参数 C 调优曲线
- `figures/lr_tuning.png`：LR 参数 C 调优曲线
- `figures/confusion_matrix.png`：最优模型验证集混淆矩阵

## 复现说明

- 随机种子固定为 42（np.random.seed 和各模型 random_state）
- 数据划分使用 stratify 分层抽样，保证类别比例一致
- TF-IDF 仅在训练子集上 fit，验证集和测试集用同一规则 transform
