# GB-0026 测试记录：多主群研究

## 环境

- 本机 Python 3.11.2；numpy 2.2.1 / scipy 1.15.1 / scikit-learn 1.6.1 / FastAPI 0.115.6。
- 本机未安装 Docker 与 PostgreSQL，因此启动路径未做 Compose 实测；用 SQLite 内存/临时文件替代验证路由、鉴权、导出与升级。PostgreSQL 专属语法仅 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`（`002_multi_group.sql`），应用内 `upgrade_schema()` 用 SQLAlchemy inspector 判断列存在后执行标准 `ALTER TABLE ... ADD COLUMN`，两种路径均被覆盖（后者有测试）。
- 前端 React 18.3.1 / TypeScript 5.7.2 / Vite 6.4.6，`npm run build` 通过（`tsc --noEmit && vite build`）。

## 命令

```sh
cd /workspace
python3 -m unittest discover -s backend/tests -v
cd frontend && npm run build
```

23 个用例全部通过：11 个算法/拒绝/兼容用例，5 个结果形状/导出列用例，7 个 API/升级用例。

## 覆盖点与结果

| 要求 | 测试 | 结果 |
|---|---|---|
| 双群：群数只由 train 决定，选 2 群 | `test_two_groups_selected_from_train_only` | 2 群，train 大小 100/100；候选 BIC/ΔBIC 记录完整 |
| 每群独立校准，p 只用同群 cal | `test_test_uses_own_group_calibration` | 每 test 的 `group_m` 等于其群 cal 数；p≥1/(m+1) |
| 冻结归属规则（最近重加权 D²，次近与间隔） | 同上 | `group_code` 为最小 D²；`assignment_margin=second−d2` |
| 行序打乱不变 | `test_row_order_and_feature_order_do_not_change_result` | fit_hash/input_hash/p/群成员完全一致 |
| 改 test 值不动 fit/cal | `test_changed_test_values_do_not_change_fit_or_calibration` | fit_hash、cal D²、群数不变；input_hash 变 |
| 改 alpha 不动评分/p | `test_changed_alpha_does_not_change_scores` | D²/p 全同；input_hash 变；筛选数变 |
| 单峰模式保留 | `test_single_mode_still_available_and_uses_one_fit` | `mode=single`，一次 MCD 拟合 |
| 历史无 `config.mode` 快照按单峰 | `test_historical_snapshot_without_config_is_single` | 与显式 single 的 fit_hash/input_hash/筛选数完全相同 |
| 小群明确拒绝，不静默丢窗 | `test_small_group_is_rejected_not_silently_dropped` | 422 文本含“小群”，列出 G1/G2 尺寸 |
| 退化训练拒绝 | `test_degenerate_training_is_rejected` | 秩亏文本，含 mode=multi 路径 |
| 某群 cal 过薄拒绝，不借他群 | `test_group_with_too_little_calibration_is_rejected` | 拒绝文本指出该群 cal 少于 4 |
| 可行性检查不读 test | `test_feasibility_never_reads_test_windows_for_selection` | 改 test 值后 feasibility 输出逐字段相等 |
| 配置默认单峰、API 校验 mode | `test_02/06` | 缺省 single；非法 mode 422 且不回显输入 |
| 结果页/下载呈现归属与依据 | `test_03` | CSV 增加 group/group_cal_m/group_assignment_basis/second_group/second_group_D_squared/assignment_margin/ambiguous/mode；未批准 403、批准后 200 |
| 角色边界 | `test_05` | viewer 创建多群 403 |
| 历史数据升级 | `test_historical_database_gains_mode_single` | 旧表无 mode 列 → 回填 `single`；重复执行幂等 |
| SL-02 训练污染多群拒绝 | `test_01` | `multi_supported=false`，原因含“小群” |
| SL-03 退化单峰拒绝 | `test_01` | `supported=false`，原因秩亏 |

## 关键样例（`multi_group_result.json`）

`mixed_windows(seed=300)`：两个近椭圆主群，train/cal/test 固定 200/150/100，行序已打乱。train BIC 在 1–5 群候选中选择 2 群（相对 1 群 ΔBIC=161.9），两群 train 各 100 条；cal 归属 G1/G2 为 73/77，test 归属为 77/73。示例测试窗口 MX-351 归属 G2，同群 cal 77 条、cal≥test 计数 19，p=(1+19)/(77+1)=0.2564；次近群 G1，D² 间隔 49.24。完整候选与逐窗归属在 `multi_group_result.json`。

## 已知差异

- 多主群 UI 只做了构建期类型检查与本地 API 联调，未在浏览器截图；本机无 Docker，未跑 Compose 健康检查。
- 群数上限 5、最小 20 条/15% 为本项目预设；更大数据需调整 `MAX_GROUPS/MIN_GROUP_SIZE/MIN_GROUP_FRACTION`。
- GMM 仍是局部最优的 EM，不以全枚举声称全局最优；BIC 只用于训练内选群。
- 多主群 `p` 是**群内**边际保证，群归属本身不是标签或危险结论；cal/test 可交换性与群内单峰假设仍须人工核对。
- 本机测试用 SQLite 代替 PostgreSQL；JSON 列在 SQLite 为文本，PostgreSQL 为 jsonb，业务层只用 Python dict 读写，未使用 PG 专属 JSON 查询。
