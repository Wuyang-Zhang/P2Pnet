# P2PNet 评估指标详解

## 概述

P2PNet (Point-to-Point Network) 是一种基于点检测的目标检测模型，主要用于人群计数、人头检测等任务。本文档详细说明了用于评估该模型性能的各种指标，包括基础回归指标、检测特定指标以及范围分析指标。

## 目录

1. [基础回归指标](#基础回归指标)
2. [检测特定指标](#检测特定指标)
3. [计数范围分析](#计数范围分析)
4. [指标使用指南](#指标使用指南)
5. [代码示例](#代码示例)

## 基础回归指标

### 1. MSE (Mean Squared Error) - 均方误差

**定义**: 预测值与真实值之间差值的平方的平均值。

**公式**:
```
MSE = (1/n) * Σ(y_true - y_pred)²
```

**特点**:
- 对大误差更敏感（平方惩罚）
- 单位与原始数据相同
- 范围: [0, +∞)

**适用场景**: 当大误差需要被严重惩罚时使用。

### 2. MAE (Mean Absolute Error) - 平均绝对误差

**定义**: 预测值与真实值之间绝对差值的平均值。

**公式**:
```
MAE = (1/n) * Σ|y_true - y_pred|
```

**特点**:
- 对所有误差一视同仁
- 直观易懂，单位与原始数据相同
- 范围: [0, +∞)

**适用场景**: 需要直观理解平均误差大小时使用。

### 3. RMSE (Root Mean Squared Error) - 均方根误差

**定义**: MSE的平方根。

**公式**:
```
RMSE = √[(1/n) * Σ(y_true - y_pred)²]
```

**特点**:
- 与MAE相比，更容易理解（同量纲）
- 对异常值敏感
- 范围: [0, +∞)

**适用场景**: 需要与MAE进行比较，且希望结果与原数据在同一量纲。

### 4. MAPE (Mean Absolute Percentage Error) - 平均绝对百分比误差

**定义**: 预测误差占真实值的百分比的平均值。

**公式**:
```
MAPE = (100/n) * Σ|(y_true - y_pred) / y_true|
```

**特点**:
- 以百分比形式表示，便于跨数据集比较
- 当真实值为0时需要特殊处理
- 范围: [0%, +∞)

**适用场景**: 需要了解相对误差大小时使用。

### 5. R² Score (Coefficient of Determination) - 决定系数

**定义**: 模型解释的方差比例。

**公式**:
```
R² = 1 - (SS_res / SS_tot)
其中 SS_res = Σ(y_true - y_pred)²
      SS_tot = Σ(y_true - y_mean)²
```

**特点**:
- 衡量模型拟合优度
- 最佳值为1，最差可为负值
- 范围: (-∞, 1]

**适用场景**: 评估模型整体拟合质量。

## 检测特定指标

### 6. 阈值内准确率 (Accuracy within Threshold)

**定义**: 预测误差在指定阈值内的样本比例。

**计算方法**:
```python
relative_error = |y_pred - y_true| / (|y_true| + ε)
accuracy = mean(relative_error <= threshold) * 100%
```

**参数**:
- `threshold`: 相对误差阈值（默认10%）

**特点**:
- 反映"基本正确"的预测比例
- 对实际应用更有参考价值
- 范围: [0%, 100%]

### 7. 精确计数准确率 (Exact Count Accuracy)

**定义**: 预测值完全等于真实值的样本比例。

**公式**:
```
Accuracy = (正确预测样本数 / 总样本数) * 100%
```

**特点**:
- 严格的准确性衡量
- 对计数任务要求很高
- 范围: [0%, 100%]

### 8. 低估率/高估率 (Underestimation/Overestimation Rates)

**定义**: 预测值低于/高于真实值的样本比例。

**计算方法**:
```python
underestimation_rate = mean(y_pred < y_true) * 100%
overestimation_rate = mean(y_pred > y_true) * 100%
```

**特点**:
- 反映模型的系统性偏差
- 有助于诊断模型问题
- 两者之和不一定等于100%（存在完全正确的预测）

### 9. 平均相对误差 (Mean Relative Error)

**定义**: 所有样本相对误差的平均值。

**公式**:
```
MRE = (100/n) * Σ|y_pred - y_true| / (|y_true| + ε)
```

**特点**:
- 整体相对误差水平
- 对不同规模的数据具有可比性
- 范围: [0%, +∞)

### 10. 误差标准差 (Error Standard Deviation)

**定义**: 预测误差的标准差。

**公式**:
```
σ = √[(1/(n-1)) * Σ(error - mean_error)²]
其中 error = y_pred - y_pred
```

**特点**:
- 反映预测的稳定性
- 较大的值表示预测不一致
- 范围: [0, +∞)

## 计数范围分析

### 11. 分范围性能分析 (Range-based Performance Analysis)

**目的**: 分析模型在不同人群密度下的表现差异。

**默认范围划分**:
- **0-10人**: 稀疏人群场景
- **10-50人**: 中等密度人群
- **50-100人**: 高密度人群
- **100+人**: 极高密度人群

**计算指标**:
- MAE: 该范围内样本的平均绝对误差
- MSE: 该范围内样本的均方误差
- RMSE: 该范围内样本的均方根误差
- 样本数量: 该范围内的样本数

**应用价值**:
- 识别模型在特定密度下的弱点
- 指导数据收集和模型优化
- 了解实际应用场景的适用性

## 指标使用指南

### 如何选择合适的指标组合

#### 1. 基础评估 (Basic Evaluation)
```python
# 建议组合
MAE, RMSE, R²
```

#### 2. 详细分析 (Detailed Analysis)
```python
# 全面评估
MAE, RMSE, MAPE, R² + 检测特定指标 + 范围分析
```

#### 3. 实际应用导向 (Application-oriented)
```python
# 实用性评估
阈值内准确率(10%) + 平均相对误差 + 范围分析
```

### 指标解读要点

#### MAE vs RMSE
- **MAE < RMSE**: 误差分布较为均匀
- **MAE ≈ RMSE**: 存在一些较大的误差
- **MAE >> RMSE**: 存在非常大的异常误差

#### 阈值内准确率
- **> 80%**: 模型性能优秀
- **60-80%**: 模型性能良好
- **40-60%**: 模型性能一般
- **< 40%**: 需要进一步优化

#### 低估/高估率
- **均衡分布**: 模型无明显系统偏差
- **一边倒**: 可能存在系统性问题
  - 高低估率 → 模型过于保守
  - 高高估率 → 模型过于激进

## 代码示例

### 基本使用

```python
from tools_usr.calculate_metric import print_comprehensive_metrics, read_data_from_file

# 读取数据
y_true, y_pred = read_data_from_file('./output/pre_gd_cnt.txt')

# 输出全面评估结果
print_comprehensive_metrics(y_true, y_pred)
```

### 单独计算特定指标

```python
from tools_usr.calculate_metric import (
    calculate_metrics,
    calculate_detection_metrics,
    calculate_counting_ranges
)

# 基础回归指标
mse, mae, rmse, r2, mape = calculate_metrics(y_true, y_pred)

# 检测特定指标
detection_metrics = calculate_detection_metrics(y_true, y_pred, threshold=0.1)

# 范围分析
range_metrics = calculate_counting_ranges(y_true, y_pred)
```

### 输出示例

```
============================================================
COMPREHENSIVE POINT DETECTION EVALUATION METRICS
============================================================

📊 BASIC REGRESSION METRICS:
MSE (Mean Squared Error): 15.6779
MAE (Mean Absolute Error): 3.9043
RMSE (Root Mean Squared Error): 3.9597
R² Score (Coefficient of Determination): 0.8765
MAPE (Mean Absolute Percentage Error): 25.43%

🎯 DETECTION-SPECIFIC METRICS:
Accuracy within 10% threshold: 68.50%
Exact count accuracy: 12.30%
Underestimation rate: 45.20%
Overestimation rate: 42.30%
Mean relative error: 28.45%
Error standard deviation: 4.1234

📈 PERFORMANCE BY COUNTING RANGES:
  Range 0-10 (n=45):
    MAE: 1.2456, RMSE: 1.5678
  Range 10-50 (n=78):
    MAE: 3.4567, RMSE: 4.1234
  Range 50-100 (n=32):
    MAE: 5.6789, RMSE: 6.7890
  Range 100-inf (n=12):
    MAE: 8.9012, RMSE: 9.4567

🔄 LEGACY METRIC:
Custom Precision (cprecision): 0.2845
============================================================
```

## 最佳实践

### 1. 定期监控
- 在训练过程中定期计算这些指标
- 关注关键指标的变化趋势
- 及时发现性能退化

### 2. 多维度分析
- 不要只依赖单一指标
- 结合业务需求选择合适的指标组合
- 关注范围分析中的薄弱环节

### 3. 基准对比
- 与现有方法进行对比
- 记录不同配置下的性能表现
- 建立性能基线

### 4. 解释结果
- MAE告诉我们"平均错多少个"
- 阈值内准确率告诉我们"多少预测是可用的"
- 范围分析告诉我们"哪些场景需要改进"

---

**注意**: 本文档描述的指标适用于基于点检测的目标检测任务，特别是人群计数、人头检测等场景。对于其他类型的目标检测任务，可能需要额外的指标如mAP、IoU等。