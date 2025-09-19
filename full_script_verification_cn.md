# 完整脚本验证说明

本文件提供逐步指导，手动验证P2P借贷数据分析项目（位于E:\Steep\STJU_SAIF\P2P_PPDAI_DATA\assignment1）中的所有脚本。目标是按顺序运行它们，检查输出、审查代码逻辑、处理错误，并理解整体工作流程。假设您在assignment1目录中使用Python环境（如PowerShell或Anaconda）运行。如果需要，安装依赖：`pip install pandas matplotlib seaborn scikit-learn xgboost tkinter`。

## 步骤1：准备（环境设置）
- **工作原理**：所有脚本依赖pandas处理数据，有些使用可视化（matplotlib/seaborn）或机器学习（sklearn/xgboost）。路径指向../P2P_PPDAI_DATA/的原始CSV文件。输出包括CSV文件（如cleaned_LC.csv、engineered_df.csv）和图表/PDF。
- **手动验证**：
  - 打开PowerShell：导航到`cd E:\Steep\STJU_SAIF\P2P_PPDAI_DATA\assignment1`。
  - 检查依赖：运行`pip list`，确保pandas、matplotlib、seaborn、scikit-learn、xgboost和tkinter已安装。如果没有，通过`pip install <package>`安装。
  - 验证原始数据：运行`dir ../P2P_PPDAI_DATA/`，确认LC.csv、LP.csv、LCIS.csv存在。
  - 边缘情况：创建小型测试CSV子集（如LC.csv的前100行），并更新脚本路径，以小数据测试速度。
  - 进度：如果设置成功，继续。预期：无缺失包或文件。

## 步骤2：验证data_load.py
- **工作原理**：定义原始CSV（LC、LP、LCIS）的灵活路径。只加载LC.csv到数据帧，打印其头/信息/形状/描述/缺失/重复。然后尝试打印未加载的LP和LCIS（这是一个bug，会引发NameError）。目的：在清理前初步检查数据，找出缺失值或重复等问题。
- **手动验证**：
  - 运行：`python data_load.py`。
  - 检查输出：查找“LC loaded successfully”，然后LC头（前5行）、信息（数据类型/计数）、形状（例如~328553行）、描述（统计）、缺失计数、重复（应为0）。预期LP/LCIS打印出错，因为数据帧未加载——临时修复：在它们的打印前添加`lp = pd.read_csv(lp_path); lcis = pd.read_csv(lcis_path)`。
  - 调试/边缘情况：使用无效路径重新运行（例如，编辑lc_path为无效值），测试try-except（应打印错误）。检查输出中的NaN。子集数据：通过`lc = pd.read_csv(lc_path, nrows=10)`运行前10行。
  - 进度：脚本成功加载/检查LC，但需修复LP/LCIS加载以完整验证。输出匹配原始数据统计吗？

## 步骤3：验证cleaning_merge.py
- **工作原理**：加载LC/LP/LCIS，从LC选择所需列，转换日期，删除关键字段中的NaN，用中位数填充数值NaN，过滤日期（2015-01-01至2017-01-30），移除异常值（年龄18-120，金额>0 <=1M，利率0-50，期限>0）。聚合LCIS（投资次数/总额，平均逾期天数）。合并所有到df（ListingId内连接，LCIS左连接）。导出cleaned_LC.csv。目的：准备干净的合并数据集用于分析。
- **手动验证**：
  - 前置条件：先运行data_load.py检查原始数据。
  - 运行：`python cleaning_merge.py`。
  - 检查输出：打印清理后LC列/形状（如果无过滤损失，预期328553行），如果行数不匹配则警告，LCIS聚合形状，合并头/形状。在Excel/Notepad中打开cleaned_LC.csv——验证行数匹配预期，无无效年龄/利率，日期已过滤。
  - 调试/边缘情况：在LC.csv的测试副本中引入NaN/异常值，重新运行，检查是否正确填充/过滤。测试合并完整性：前后计数唯一ListingId（应匹配内连接逻辑）。如果形状错误，打印df.isnull().sum()。
  - 进度：打印合并DF形状，cleaned_LC.csv创建且有效吗？

## 步骤4：验证exploratory.py
- **工作原理**：加载cleaned_LC.csv和原始LP/LCIS，类似cleaning_merge.py进行清理/合并（包含LCIS聚合）。计算逾期统计（值计数），绘制分布（借款金额）、逾期细分、相关热图。分组平均（例如，按标签的金额）。添加LCIS统计/图表。目的：可视化探索性数据分析，理解数据分布、相关性和逾期模式。
- **手动验证**：
  - 前置条件：运行cleaning_merge.py生成cleaned_LC.csv。
  - 运行：`python exploratory.py`。
  - 检查输出：逾期统计（比例）、按标签的平均金额、num_investments统计。打开保存的图表（distribution.png、overdue_breakdown.png、correlations.png、num_investments_dist.png）——验证它们匹配数据（例如，直方图峰值、热图颜色表示相关性）。
  - 调试/边缘情况：如果图表为空，检查df[label_col]的有效值（0/1）。子集测试：在图表前添加`df = df.sample(100)`，重新运行。在Excel中手动计算2-3列的相关性验证。
  - 进度：图表生成且可打开，统计合理（例如，逾期率~某个%）吗？

## 步骤5：验证feature_engineering.py
- **工作原理**：加载cleaned_LC.csv和原始LP/LCIS，类似先前脚本进行清理/合并。工程特征：payment_ratio（已付/到期本金）、total_overdue_periods（从LP求和）、overdue_rate（历史逾期/总期数），独热编码分类变量（性别、评级、类型）。保存engineered_df.csv。目的：从原始数据创建预测特征用于建模。
- **手动验证**：
  - 前置条件：运行cleaning_merge.py。
  - 运行：`python feature_engineering.py`。
  - 检查输出：新特征头（例如，payment_ratio ~1表示全额支付）、工程DF形状（应添加如性别虚拟列）。
  - 打开engineered_df.csv：抽查计算（例如，对于一行，手动计算overdue_rate = 历史逾期 / (正常 + 逾期)）。
  - 调试/边缘情况：测试零除（例如，总期数=0的行，应为1）。引入坏数据（负本金），重新运行查看处理。验证虚拟列：计数新列（例如，初始评级有多个级别 → 多个虚拟）。
  - 进度：engineered_df.csv创建，特征计算正确吗？

## 步骤6：验证modeling.py
- **工作原理**：加载engineered_df.csv，二值化标签（>0为逾期），独热编码剩余分类变量，分层拆分70/30，计算类别权重处理不平衡，训练XGBoost，预测，打印测试准确率和5折CV准确率。目的：构建基线模型预测逾期贷款。
- **手动验证**：
  - 前置条件：运行feature_engineering.py。
  - 运行：`python modeling.py`。
  - 检查输出：测试准确率（例如，>0.7?）、CV平均/标准差（低标准差=稳定）。
  - 调试/边缘情况：子集df到1000行加速，重新运行——准确率应类似。测试不平衡：打印y.value_counts()预拆分。如果label_col错误，手动设置为'还款状态'。交叉验证：保存preds到CSV，手动检查几行。
  - 进度：模型无错误训练，准确率打印吗？

## 步骤7：验证evaluation.py
- **工作原理**：加载engineered_df.csv，类似modeling.py准备/拆分/训练。计算预测/概率、指标（准确率/精确率/召回率/F1/ROC-AUC），绘制ROC曲线，运行5折CV。目的：超出基本准确率的彻底模型评估。
- **手动验证**：
  - 前置条件：运行feature_engineering.py。
  - 运行：`python evaluation.py`。
  - 检查输出：所有指标（例如，ROC-AUC >0.8?）、CV分数。打开roc.png——曲线应在上对角线以上。
  - 调试/边缘情况：使用完美数据测试（所有标签=0）→预期高准确率但低召回。与modeling.py比较指标（应一致）。
  - 进度：指标打印，ROC图有效吗？

## 步骤8：验证summarize.py
- **工作原理**：加载engineered_df.csv，计算逾期率和与标签的前相关。硬编码模型指标（占位符——手动更新）。写入markdown summary.md，包括发现、教训、改进。目的：将洞见编译成报告。
- **手动验证**：
  - 前置条件：运行evaluation.py获取真实指标（编辑summarize.py使用它们代替占位符）。
  - 运行：`python summarize.py`。
  - 检查输出：打印逾期率/相关。打开summary.md——验证内容匹配（例如，相关降序排序）。
  - 调试/边缘情况：如果label_col错误，修复它。小df测试——统计应缩放。
  - 进度：summary.md生成且可读吗？

## 步骤9：验证gui.py
- **工作原理**：Tkinter GUI，按钮加载数据（合并cleaned_LC + LP）、显示图表（逾期/相关）、训练模型（子集带滑块test_size）、通过子进程运行评估/总结。自动打开文件。目的：为非编码者提供交互界面。
- **手动验证**：
  - 运行：`python gui.py`（打开窗口）。
  - 交互：点击“Load Cleaned Data”→成功消息。“Show Overdue Breakdown Plot”→图表打开。调整滑块，“Train XGBoost Model”→准确率弹出。“Run Model Evaluation”→指标弹出。“Generate and View Summary”→summary.txt打开。
  - 调试/边缘情况：无cleaned_LC.csv测试→错误弹出。关闭/重新打开GUI，重复。如果子进程失败，检查路径。
  - 进度：所有按钮工作，弹出/图表出现吗？

## 步骤10：整体工作流程检查
- 端到端运行：按顺序执行所有脚本（修复如data_load.py中的bug）。检查链式输出（例如，engineered_df.csv用于建模）。
- 边缘情况：使用10%数据子集全流程运行——验证一致性。引入错误（例如，删除CSV中的列）并调试。
- 验证集成：全运行后，检查summary.md是否反映评估指标。

此文件覆盖完整项目工作流程。如果出现问题，请分享错误日志以修复！
