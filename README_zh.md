# 可配置的 PU Bagging 框架

将原来的 RF + GA 单脚本整理为独立模块。研究对象仍是：在缺少可靠负样本时，背景样本选择怎样影响预测和特征解释，以及怎样通过重复抽样和汇总获得稳定结果。RF、XGBoost等是可更换的模型，GA、Optuna等是可更换的参数搜索工具。

## 一次完整运行的顺序

1. 选择数据集和特征，确定正样本P与符合纳入条件的未标记背景池U。
2. 选择模型、预处理方式和超参数范围。
3. 选择优化器；每组候选参数用固定数量的重复抽样与交叉验证评分。
4. 确定最优参数及抽样比例后，再按**期望覆盖率**计算最终轮数。
5. 用新抽样进行重复交叉验证，记录每轮、每折AUC。
6. 用全部正样本和每轮背景训练子模型，汇总区域预测均值和子模型间标准差。
7. 可选计算各轮SHAP及交互值，汇总重要性和等频分箱统计。
8. 按配置输出表格、图件、GeoTIFF和可重复使用的模型。

## 安装与运行

在项目文件夹打开终端：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all,dev]"
python examples/make_demo_data.py
python -m pu_bagging validate examples/demo.yaml
python -m pu_bagging run examples/demo.yaml
```

不启用GIS、SHAP等模块时，可以只安装 `python -m pip install -e ".[plots]"` 运行基础示例。示例数据为合成数据，不含论文原始数据。

完整输出示例：

```powershell
python -m pu_bagging run examples/full_outputs.yaml
```

每次运行使用新的 `output_dir`，避免覆盖已有结果。所有相对路径均相对于YAML文件所在目录。

## 换模型、优化器和数据

| 要修改的内容 | 配置位置 | 示例 |
|---|---|---|
| 模型 | `model.name` | `random_forest`、`extra_trees`、`logistic_regression`、`xgboost` |
| 固定模型参数 | `model.params` | `n_estimators: 200` |
| 参数优化算法 | `optimizer.name` | `genetic`、`random`、`grid`、`optuna`、`none` |
| 搜索范围 | `optimizer.space` | `model.max_depth`、`sampling.ratio`、`training.positive_weight` |
| 正负比例与轮数 | `sampling` | 比例、期望覆盖率或指定轮数 |
| 数据 | `data` | CSV或栅格+样本点 |
| 交叉验证 | `cv` | 分层、分组、空间区块 |
| 训练特征缩放 | `preprocess` | 某特征乘1e12，或训练折内标准化 |
| SHAP | `explain` | 是否启用、解释方法、等频分箱、交互分析 |
| 表格/栅格输出 | `outputs` | CSV、Excel、GeoTIFF分别开关 |
| 图件 | `plots` | 各图类型开关、格式、分辨率 |

CSV必须有唯一ID；`label=1`表示正样本，`label=0`表示**未标记背景**，不是已经确认无资源的位置。可以另提供预测网格CSV。字段名称不要求固定，在YAML中对应即可。

GIS数据使用 `examples/raster_template.yaml`：所有预测栅格须同CRS、同范围、同分辨率和网格起点；点数据会转换到栅格CRS。空间区块大小按投影坐标单位设置。框架不自动猜测正样本缓冲区半径；请先准备合适的未标记候选点池。

`examples/original_study_template.yaml`额外保留了原脚本20个特征的名称、栅格文件名、参数范围和20次初筛设置，便于迁移当前研究。路径使用可替换的相对路径；原文件名`Vs30m_clip1.tif`按原脚本保留。

自定义模型可以写成 `model.name: my_package:MyClassifier`，只要兼容sklearn并实现 `fit`、`predict_proba` 即可；不用改PU主流程。模型类型不限于树，非树模型的SHAP选择 `permutation`。自定义优化器也可使用 `module:function`，见[扩展接口](docs/configuration.md)。

## 图件和结果模块

- 每轮AUC散点及均值线、平均ROC及波动范围、参数搜索历史。
- 预测均值图、子模型间标准差图；栅格输入额外导出保留地理信息的GeoTIFF。
- 特征相关矩阵和已知正样本累计捕获曲线。
- SHAP重要性均值/标准差图；重要性分布图以箱体表示25%–75%分位数、竖线表示均值、须表示均值±标准差。
- SHAP依赖图以蓝色折线表示分箱均值，阴影分别表示最小–最大值、25%–75%分位数及均值±标准差，与论文图8、图9的统计含义一致。
- 可选SHAP交互矩阵和Top交互对依赖图。
- 所有绘图依据表格可导出CSV和Excel；每轮SHAP原始结果存为NPZ。
- 可保存每个子模型的完整预处理和模型管线，之后调用 `predict` 对新CSV预测。

默认600 ppi，支持PNG、PDF、SVG和TIFF。依赖图默认20个分位箱，去除两端各1%；重复值导致边界重合时箱数会减少，最大值纳入最后一箱，以分箱上下界的中点作为横坐标，原始物理量用于横轴。

已有结果可以独立重绘，无需重新训练或读取原始研究数据：

```powershell
python -m pu_bagging plot outputs/full_outputs outputs/replot --dpi 600 --formats png pdf
python -m pu_bagging plot outputs/full_outputs outputs/replot_shap --only shap_importance shap_dependence --formats svg
```

重绘读取导出的CSV（或Excel）和`plot_context.json`。输出到新目录，不覆盖旧图。

## 与原脚本有关的关键定义

- 先优化比例再决定抽样次数。每轮不放回抽样，跨轮允许重复。
- 覆盖率公式给出的是期望覆盖率；结果同时记录实际覆盖率。
- 默认固定子模型随机种子，不再用当前时间产生种子。每次同数据、同配置、同依赖环境可以复现。
- 交叉验证是参数选择后的内部评估，不把它称为独立外部验证或嵌套交叉验证。
- AUC标签是“已知正样本/临时负标签背景”，不是已确认的资源存在/不存在。
- 预测标准差是子模型输出波动，不是均值标准误或已校准置信区间。
- SHAP汇总基于各轮训练集；同一正样本或背景被多次解释时会重复纳入。可用 `max_samples` 限制计算量，此时会记录实际解释行数。
- Tree SHAP按树模型的原生输出解释：RF通常是概率，XGBoost通常是原始间隔值，不能直接跨模型比较绝对SHAP大小。

更详细的差异、原变量与新配置的映射见[迁移说明](docs/migration.md)。

已对照论文方法与原始脚本完成[方法逻辑核对](docs/method_consistency.md)。框架保留单模型PU Bagging完整流程；按作者要求，不包含两模型共识分区，也不把论文的最优参数、轮数或AUC写死。

## 上传GitHub

上传项目源码目录或提供的ZIP源码包。不要上传 `.venv`、`.test-env`、`data`、`outputs`。这些目录已经写入 `.gitignore`。准备好的源码包不含论文数据和本地运行结果。

当前没有代作者选择许可证。正式公开前可自行添加MIT、BSD等适合的许可证；论文发表后再补正式引用信息。

源码打包可运行 `python scripts/package_source.py`。本地验证结果见[验证记录](docs/validation.md)。
