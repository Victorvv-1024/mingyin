# 销售预测管道 - 完整使用指南

[English](USAGE_GUIDE.md) | **中文简体**

本综合指南涵盖销售预测管道使用的各个方面，从基础使用到高级自定义。

## 📋 目录

1. [快速开始](#快速开始)
2. [管道概览](#管道概览)
3. [阶段1：特征工程](#阶段1特征工程)
4. [阶段2：模型训练](#阶段2模型训练)
5. [阶段3：模型评估](#阶段3模型评估)
6. [模型比较](#模型比较)
7. [高级配置](#高级配置)
8. [输出分析](#输出分析)
9. [故障排除](#故障排除)
10. [最佳实践](#最佳实践)

## 🚀 快速开始

### 最小设置

```bash
# 1. 准备您的数据
mkdir -p data/raw
# 放置您的Excel文件：2021.xlsx、2022.xlsx、2023.xlsx

# 2. 运行增强模型的完整管道
python src/scripts/run_complete_pipeline.py \
    --data-dir data/raw \
    --output-dir outputs \
    --model-type enhanced \
    --run-phase3

# 3. 检查结果
ls outputs/*/enhanced_models/  # 模型文件
ls outputs/*/2023_evaluation/  # 评估结果
```

### 完整功能演示

```bash
# 训练两个模型并比较性能
python src/scripts/run_complete_pipeline.py \
    --data-dir data/raw \
    --output-dir outputs \
    --years 2021 2022 \
    --epochs 120 \
    --batch-size 256 \
    --model-type both \
    --run-phase3 \
    --experiment-name "生产比较" \
    --random-seed 42
```

## 🏗️ 管道概览

### 三阶段架构

```
原始Excel数据 → 阶段1：特征工程 → 工程化数据集
                        ↓
           阶段2：模型训练 → 原版模型 + 增强模型
                        ↓
           阶段3：评估 → 2023年性能结果 + 模型比较 + 业务建议
```

### 模型架构比较

| 特征 | 原版嵌入模型 | 增强嵌入模型 |
|---------|----------------------|------------------------|
| **目的** | 基线比较 | 生产部署 |
| **架构** | 标准嵌入+注意力 | 高级多头注意力+残差 |
| **优化器** | Adam | AdamW学习率调度 |
| **正则化** | 基础dropout | 高级dropout+批归一化 |
| **性能** | ~18-22% MAPE | ~14-18% MAPE |
| **训练时间** | ~70分钟 | Mac上~24小时 |

## 📊 阶段1：特征工程

### 概览

阶段1将原始Excel数据转换为针对中国电商销售预测优化的复杂特征集。

### 独立运行阶段1

```bash
python src/scripts/phase1_feature_engineering.py \
    --data-dir data/raw \
    --output-dir outputs/engineered \
    --years 2021 2022 2023
```

### 创建的特征类别（总计316个特征）

#### 1. 🕒 时间特征（24个）
- **中国日历整合**：春节、国庆黄金周、双十一、中秋节
- **季节模式**：月份编码、季度编码、节假日月份、季节编码
- **滞后特征**：1-12个月滞后值

#### 2. 🎯 促销特征（12个）
- **事件检测**：促销期间、假期季节
- **平台特定促销**：抖音、京东、天猫促销交互
- **响应度分级**：高/中/低促销响应类别

#### 3. 📊 滞后特征（15个）
- **多周期滞后**：销售、金额、价格的1、2、3、6、12个月滞后

#### 4. 📈 滚动统计（24个）
- **窗口大小**：3、6、12个月滚动
- **统计指标**：均值、标准差、最小值、最大值、变异系数、四分位数

#### 5. ⚡ 动量特征（16个）
- **同比分析**：上升/下降/稳定趋势
- **加速度指标**：销售加速度、波动性
- **动量分类**：动量方向和持续时间

#### 6. 🏪 客户行为分析（124个）
- **店铺分析**：性能指标、多样性、质量指标、店铺类型
- **品牌分析**：市场表现、分销指标、定价策略
- **平台分析**：市场份额、多平台策略
- **中国白酒品牌**：茅台、五粮液、西凤、习酒等专项追踪

#### 7. 🏆 店铺分类（4个）
- **竞争定位**：领导者、主要、次要、利基

#### 8. 🔄 平台动态（19个）
- **跨平台竞争**：多平台存在、性能比率
- **客户行为**：平台忠诚度、切换模式
- **经验级别**：新手、有经验、资深

#### 9. 🚨 峰值检测（5个）
- **异常检测**：销售z分数、峰值分类、偏差分析

## 🤖 阶段2：模型训练

### 运行模型训练

```bash
# 仅训练增强模型
python src/scripts/phase2_model_training.py \
    --engineered-dataset outputs/engineered/dataset.pkl \
    --output-dir outputs/models \
    --model-type enhanced \
    --epochs 100

# 训练两个模型进行比较
python src/scripts/phase2_model_training.py \
    --engineered-dataset outputs/engineered/dataset.pkl \
    --output-dir outputs/models \
    --model-type both \
    --epochs 100
```

### 训练配置

```yaml
# 推荐设置
epochs: 100-150
batch_size: 256-512
learning_rate: 0.001
early_stopping: true
patience: 10
```

## 🎯 阶段3：模型评估

### 2023年测试评估

```bash
python src/scripts/phase3_test_model.py \
    --models-dir outputs/models \
    --engineered-dataset outputs/engineered/dataset.pkl \
    --test-year 2023
```

### 评估指标

- **MAPE（平均绝对百分比误差）**：主要性能指标
- **RMSE（均方根误差）**：预测精度
- **R²分数**：模型拟合度
- **业务指标**：店铺、品牌、平台表现分析

## 📈 输出分析

### 模型比较报告

```
outputs/{实验名称}/reports/
├── 📄 model_comparison.json     # 模型比较指标
├── 📄 training_summary.txt      # 训练结果总结
├── 📄 performance_plots.png     # 性能可视化
└── 📄 business_insights.txt     # 业务见解
```

### 关键性能指标

| 指标 | 原版模型 | 增强模型 | 目标 |
|------|----------|----------|------|
| 验证MAPE | ~10-22% | ~14-18% | <20% |
| 2023测试MAPE | ~30-45% | ~16-20% | <20% |
| 训练时长 | ~70分钟 | Mac上~24小时 | - |

## 🔧 高级配置

### 特征工程配置

在`src/config/feature_config.yaml`中自定义特征创建：

```yaml
# 时间特征设置
temporal_features:
  lag_periods: [1, 2, 3, 6, 12]
  rolling_windows: [3, 6, 12]
  enable_chinese_calendar: true

# 业务特征设置
business_features:
  enable_store_categorization: true
  enable_platform_dynamics: true
  store_tier_threshold: [100000, 500000]
```

### 模型参数调优

```bash
# 高级参数设置
python run_complete_pipeline.py \
    --epochs 150 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --dropout-rate 0.3 \
    --l2-regularization 0.01 \
    --early-stopping-patience 15
```

## ❗ 故障排除

### 常见问题

1. **内存不足错误**
   ```bash
   # 减少批量大小
   --batch-size 128
   ```

2. **训练时间过长**
   ```bash
   # 减少epochs或使用GPU
   --epochs 50
   ```

3. **数据格式错误**
   - 检查Excel文件编码（建议UTF-8）
   - 验证列名是否正确
   - 确保日期格式一致

### 日志检查

```bash
# 查看详细日志
tail -f logs/pipeline_YYYYMMDD.log

# 检查错误
grep ERROR logs/pipeline_YYYYMMDD.log
```

## 💡 最佳实践

### 数据准备

1. **数据质量**：确保数据完整性和一致性
2. **编码格式**：使用UTF-8编码处理中文字符
3. **日期格式**：统一日期格式（YYYY-MM-DD）

### 模型训练

1. **硬件要求**：推荐使用GPU加速训练
2. **实验管理**：使用有意义的实验名称
3. **版本控制**：记录模型版本和参数

### 性能优化

1. **批量大小**：根据可用内存调整
2. **早停**：防止过拟合
3. **学习率调度**：使用自适应学习率

### 生产部署

1. **模型选择**：优先使用增强模型
2. **性能监控**：定期评估模型性能
3. **重训练策略**：根据新数据定期重训练

---

**📞 技术支持**：如有问题，请查看GitHub问题页面或联系开发团队。 