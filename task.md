## 1. 项目目标（Project Objectives）

基于中国城市尺度碳排放数据（1992–2023）与城市社会经济统计数据（2002–2023），成以下目标：

1. **揭示碳排放变化趋势**
   - 计算全国层面与城市层面的碳排放时间序列趋势
   - 分析区域差异、增长/下降阶段、关键拐点
   - 绘制可直接出版的趋势图、空间分布图、变动速率图
2. **识别碳排放的社会经济驱动因素**
   - 选取 GDP、人口、产业结构、能源结构、城镇化率等变量
   - 进行统计分析（相关性、回归、分解模型等）
   - 通过 Kaya 恒等式或 LMDI 分解 CO₂ 增长来源
   - 构建驱动解释报告
3. **如果进行预测（可选）**
   - 使用机器学习、时间序列或多元回归预测未来碳排放趋势
   - 输出模型效果评估指标（如 RMSE、MAE、R²）
4. 生成最终的生态环境形势分析报告，用于决策支持。

## 2. 数据准备（Data Preparation）

### 2.1 数据源

| 数据集 | 文件路径 | 时间范围 | 内容说明 |
| --- | --- | --- | --- |
| 城市 CO₂ 排放数据 | `./input/city-level CO2 emissions from 1992 to 2023.csv` | 1992–2023 | 329 城市 CO₂ 排放量，由 NTL + XGBoost + MTL 校正估算 |
| 城市统计数据 | `./input/statistics_yearly.csv` | 2002–2023 | GDP、人口、产业结构、能源消费、投资、城镇化率等 |

### 2.2 数据清洗要求

1. **统一城市名称、代码、行政区划（必做）**
   - 使用城市代码作为主键
   - 处理 2000 年以来行政区划变更（如地级市设立、合并等）
   - 检查两份数据的匹配率，并生成匹配日志
2. **时间对齐**
   - 1992–2001 年仅用碳排放
   - 2002–2023 年进行全变量分析
   - 数据按城市-年份双索引合并
3. **缺失值处理**
   - 连续变量：线性插值或相邻年度填充
   - 分类变量：最近年度填充
   - 生成缺失率报告
4. **变量构建（必需）**
   - 人均 GDP
   - 单位 GDP 排放强度（CO₂/GDP）
   - 产业结构比例（第一、第二、第三产业）
   - 城镇化率
   - 能源结构 proxy（如第二产业比重）
   - 排放年增长率
5. 清洗后的数据应统一存入 `./output/data/cleaned_data.csv`。

## 3. 输出结果（Outputs）

本项目需生成以下三大类结果。

### 3.1 图表输出（`./output/figs/`）

所有图表需满足以下标准：
- 使用 matplotlib
- 能在出版物中直接使用（高 DPI、布局规整、科学可视化风格）
- 支持中文（字体需设置为 SimHei 或其他可用中文字体）
- 统一色系、统一标注风格

需生成的图表清单（建议）：
1. 全国 CO₂ 排放趋势图（1992–2023） → `./output/figs/national_trend.png`
2. 分城市 CO₂ 排放变化热图（heatmap） → `./output/figs/city_heatmap.png`
3. 城市排放空间分布图（按省汇总或地图） → `./output/figs/spatial_distribution.png`
4. 主要社会经济指标与 CO₂ 的相关矩阵图 → `./output/figs/correlation_matrix.png`
5. 排放强度（CO₂/GDP）变化图 → `./output/figs/emission_intensity_trend.png`
6. LMDI 或 Kaya 分解贡献柱状图 → `./output/figs/LMDI_contribution.png`
7. 回归分析结果图（可选） → `./output/figs/regression_results.png`
8. 模型预测曲线（如使用学习模型） → `./output/figs/model_prediction.png`

### 3.2 最终研究报告（`./output/reports/report.md`）

报告需使用 Markdown，可直接转为 PDF 或发布文章。建议结构如下：

#### 📄 生态环境形势分析报告（模板结构）

**一、形势特征（Situational Characteristics）**：全国 CO₂ 排放总体演变、城市群排放特征、排放强度变化、产业结构关联性、区域差异与空间格局、主要驱动因素（基于 LMDI 或回归结果）。

**二、形势判断（Situation Assessment）**：识别压力最大的地区、下降趋势地区、碳达峰迹象、未来排放走势、城市群对全国碳排放的决定性作用。

**三、对策建议（Policy Recommendations）**：产业结构调整、低碳城市群发展路径、区域差异化碳减排策略、科技创新、城镇化与交通结构改进、能源结构调整与绿色金融政策。

最终文件路径固定为 `./output/reports/report.md`。

### 3.3 模型预测（如执行）

若开展预测任务，需要额外输出：
- 模型预测文件：`./output/models/model_predictions.csv`
- 模型评估文件：`./output/models/model_evaluation.json`，需包含 RMSE、MAE、R² 指标及训练/测试拟合结果（同时输出拟合图）。

## 4. 最终目录结构

```
output/
 ├── data/
 │     └── cleaned_data.csv
 ├── figs/
 │     ├── national_trend.png
 │     ├── city_heatmap.png
 │     ├── spatial_distribution.png
 │     ├── correlation_matrix.png
 │     ├── emission_intensity_trend.png
 │     ├── LMDI_contribution.png
 │     └── model_prediction.png
 ├── models/
 │     ├── model_predictions.csv
 │     └── model_evaluation.json
 └── reports/
       └── report.md
```
