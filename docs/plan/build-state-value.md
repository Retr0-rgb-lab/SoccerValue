# 状态价值模型构建计划

> 给实现者：按任务顺序做。每项都有自己的测试，做完一项再做下一项。不要改传球成功率网络的输出，不要训练世界模型。仓库测试用 `python -m unittest`，不要改成 pytest。未经要求不要提交 git。

**目标：** 在真实比赛的控球时刻上，训练一个只看当前局面的状态价值 \(V(s,T)\)，并回答：加上双方站位之后，对未来 10 秒射门机会的预测是否比「球在哪、谁控球」更好。

**架构：** 先把整场事件留成时间线，再在动作完成点抽样，用该时刻之前的追踪帧作输入，用之后 10 秒内的 `ShotAtGoal` 作两头标签。常数和逻辑回归先建立对照，小 CNN 只多看站位。\(V\) 是两个概率之差。\(Q(s,a)\) 和世界模型采样不在本计划里实现。

**技术栈：** 现有 `lxml`、`numpy`、`scikit-learn`、`torch`。比赛划分用 `soccervalue/matches.py` 里已经写死的 5 场训练、1 场验证、1 场测试。

## 全局约束

- 标签球队 \(T\) 在抽样时固定。之后 10 秒里本方射门和对方射门都相对这个 \(T\)，被断球之后也不改成「谁拿球谁得正分」。
- \(V_{\text{shot}}=\hat p_{\text{for}}-\hat p_{\text{against}}\) 是射门机会代理。输出和文件名里不要写成 EPV 或进球概率。
- 主实验窗口是 10 秒。5 秒和 20 秒只用同一套代码另跑，三个窗口的损失不要加在一起。
- 窗口碰到 `FinalWhistle`、下一次开球，或没有可用追踪帧，样本丢弃，不要记成「没有射门」。
- 特征只用该时刻及之前的追踪。未来事件只用来写标签。
- 状态帧的同步只允许落在事件时刻或它之前。不要复用传球实验里会选到事件之后的那套代价。
- 栅格通道是「球队 \(T\) / 对方 / 球」，\(T\) 的进攻方向朝 \(+x\)。不要沿用「当前控球方就是进攻方」。
- 划分保持 `TRAIN_MATCHES` / `VAL_MATCHES` / `TEST_MATCHES`。验证集只给 CNN 早停，不拿测试集选模型。
- 结果写到 `outputs/state_value/`，不要覆盖 `outputs/pass_success/`。
- 相邻帧不是独立样本。抽样只发生在动作完成点。

## 文件

- 修改 `soccervalue/events.py`：新增整场时间线，保留现有 `load_open_flat_passes`。
- 新建 `soccervalue/state_value.py`：抽样、截断、两头标签。
- 新建 `soccervalue/state_frames.py`：只向过去对齐追踪帧，并转到 \(T\) 的坐标系。
- 新建 `soccervalue/value_model.py`：简单特征、两头逻辑回归、两头 CNN。
- 新建 `soccervalue/run_state_value.py`：训练、10 秒主结果、5/20 秒敏感性、失败样例。
- 新建 `tests/test_state_value.py`。
- 不改 `soccervalue/model.py` 的成功率头，不改 `soccervalue/run_pass_success.py` 的训练目标。

---

### 任务 1：留下整场事件时间线

**文件：**

- 修改 `soccervalue/events.py`
- 测试 `tests/test_state_value.py`

**产生：** `load_timeline(events_path, match_id) -> list[dict]`，按时间排序。每条至少有 `time`、`event_id`、`event_tag`、`team_id`、`player_id`、`x`、`y`、`game_section`、`is_shot`、`is_goal`、`xg`、`possession_change`。

**规则：**

- `ShotAtGoal`：`is_shot=True`，`team_id` 用该节点的 `Team`。子节点是 `SuccessfulShot` 时 `is_goal=True`。有 `xG` 就记成浮点数。
- `KickOff`：记下 `GameSection` 和 `TeamLeft`。之后的事件沿用这个 `game_section`，直到下一个 `KickOff` 或 `FinalWhistle`。
- `FinalWhistle`：`event_tag` 为 `FinalWhistle`，`game_section` 用它自己的属性。
- `Play`：`team_id`、`player_id`、坐标、`evaluation`。
- `TacklingGame`：`possession_change` 来自 `PossessionChange`；`team_id` 用 `WinnerTeam`。
- `OtherBallAction`：有 `Team` 和坐标就保留。
- 坐标优先 `X-Position` / `Y-Position`，没有则用 `X-Source-Position` / `Y-Source-Position`。
- `load_open_flat_passes` 的过滤结果保持不变，现有 `tests/test_pass_success.py` 必须仍然通过。

- [ ] **步骤 1：写会失败的测试**

在 `tests/test_state_value.py` 写一个小 XML：上半场开球、一次成功传球、一次 `ShotAtGoal` 且子节点为 `SuccessfulShot` 并带 `xG="0.2"`、一次 `PossessionChange="true"` 的 `TacklingGame`、`FinalWhistle`。

```python
def test_timeline_keeps_shot_goal_and_half_boundary():
    events = load_timeline(path, "M1")
    by_id = {item["event_id"]: item for item in events}
    self.assertTrue(by_id["shot"]["is_shot"])
    self.assertTrue(by_id["shot"]["is_goal"])
    self.assertAlmostEqual(by_id["shot"]["xg"], 0.2)
    self.assertEqual(by_id["whistle"]["event_tag"], "FinalWhistle")
    self.assertEqual(by_id["shot"]["game_section"], "firstHalf")
    self.assertTrue(by_id["tackle"]["possession_change"])
```

- [ ] **步骤 2：运行并确认失败**

`python -m unittest tests.test_state_value.TimelineTests.test_timeline_keeps_shot_goal_and_half_boundary -v`

预期：`load_timeline` 不存在。

- [ ] **步骤 3：实现 `load_timeline`**

解析一次 XML。不要在 `load_open_flat_passes` 里把非传球事件丢掉之后再无法找回。传球函数可以继续调用自己的过滤，但时间线函数必须看到射门和哨声。

- [ ] **步骤 4：两条测试都通过**

`python -m unittest tests.test_pass_success tests.test_state_value -v`

---

### 任务 2：从时间线构造状态样本

**文件：**

- 新建逻辑放在 `soccervalue/state_value.py`
- 测试 `tests/test_state_value.py`

**消费：** `load_timeline` 的列表。

**产生：** `build_state_samples(events, window_s=10.0) -> list[dict]`

每条样本：

- `time`、`event_id`、`match_id`、`team_id`（这就是 \(T\)）
- `x`、`y`、`player_id`
- `possession_by_t`：0 或 1
- `y_for`、`y_against`：0 或 1
- `window_s`

**抽样：**

- `Play` 且有球队和坐标：\(T\) 是 `Play` 的球队。`evaluation` 属于成功时 `possession_by_t=1`，属于失败时 `possession_by_t=0`。
- `OtherBallAction` 且有球队和坐标：\(T\) 是该球队，`possession_by_t=1`。
- `TacklingGame` 且 `possession_change` 为真：\(T\) 是 `WinnerTeam`，`possession_by_t=1`。没有发生球权变化的争抢不抽样。
- 不把 `ShotAtGoal` 本身当作状态。射门已经发生，不能再用它预测「会不会射门」。

**标签：**

- 只看 \((t, t+\text{window}]\) 里的 `ShotAtGoal`。等于 \(t\) 的事件不算。
- 射门球队等于 \(T\) 则 `y_for=1`。其他球队的射门则 `y_against=1`。同一窗口两边可以同时为 1。
- 若 \(t\) 与下一条 `FinalWhistle` 或下一条 `KickOff` 的间隔小于 `window_s`，丢弃该样本。
- 样本必须和边界落在同一个 `game_section`。

- [ ] **步骤 1：写测试**

构造这条时间线：第 0 秒 \(T\) 成功传球；第 6 秒 \(T\) 射门；第 12 秒对方射门；第 30 秒对方成功控球且之后没有射门；第 35 秒上半场哨声；第 36 秒一次距哨声只有 3 秒的传球。

```python
def test_labels_use_the_fixed_team_and_drop_censored_windows():
    samples = build_state_samples(events, window_s=10)
    by_id = {item["event_id"]: item for item in samples}
    self.assertEqual((by_id["pass0"]["y_for"], by_id["pass0"]["y_against"]), (1, 0))
    self.assertEqual((by_id["opp"]["y_for"], by_id["opp"]["y_against"]), (0, 0))
    self.assertNotIn("late", by_id)
```

再测一次失败传球：`possession_by_t==0`，但 8 秒后对方射门仍使 `y_against==1`，\(T\) 仍是原传球方。

- [ ] **步骤 2：确认测试失败，然后实现 `build_state_samples`**
- [ ] **步骤 3：`python -m unittest tests.test_state_value -v` 通过**

---

### 任务 3：状态帧只取事件当时或更早

**文件：**

- 新建 `soccervalue/state_frames.py`
- 测试 `tests/test_state_value.py`

**消费：** 带 `time`、`x`、`y`、`player_id`、`team_id` 的样本，以及 `soccervalue.tracking._read_trajectories`。

**产生：**

- `attach_state_frame(tracking_path, samples, pitch) -> None`，就地写入 `track_ball`、`track_players`、`frame_signed_dt`。
- `orient_state(sample, transform, pitch_x, pitch_y) -> dict | None`

**对齐：**

- 只在 \([t-6\text{ 秒}, t]\) 里找帧。`frame_signed_dt = 帧时间 - 事件时间`，结果必须 `<= 0`。
- 在这些帧里沿用传球实验的球距和持球人距离代价，选代价最小的一帧。找不到就留空，调用方丢弃。
- 不要调用 `attach_tracking`。那个函数会选到事件之后的帧。

**坐标：**

- 用 `choose_transform` 在训练集上选一次平移，验证和测试沿用这一个名字。
- `team_id == team_left` 时不翻转；否则 \(x,y\) 都按球场尺寸对称翻转，使 \(T\) 朝 \(+x\)。
- 返回 `ball`、`team`（\(T\) 的球员点）、`opponent`、`possession_by_t`、`y_for`、`y_against`、`pitch`、`match_id`、`event_id`。
- 球和事件坐标在变换后相距超过 8 米则返回 `None`。

- [ ] **步骤 1：测试过去帧优先**

两个候选帧：事件前 0.4 秒且球距 2 米；事件后 0.1 秒且球距 0.2 米。`attach_state_frame` 必须选前一帧。

```python
def test_state_frame_does_not_look_after_the_event():
    self.assertLessEqual(sample["frame_signed_dt"], 0.0)
```

- [ ] **步骤 2：测试朝向**

\(T\) 不是左攻球队时，球的 \(x\) 变成 `pitch_x - x`，\(T\) 的球员进 `team`，对方进 `opponent`。

- [ ] **步骤 3：实现并跑通 `tests.test_state_value`**

---

### 任务 4：常数和简单状态模型

**文件：**

- `soccervalue/value_model.py`
- 测试 `tests/test_state_value.py`

**消费：** `orient_state` 的字典。

**产生：**

- `simple_features(rows) -> tuple[np.ndarray, np.ndarray, np.ndarray]`，两列标签是 `y_for`、`y_against`。
- `SimpleStateValue`：一个 `LogisticRegression` 对两头分别拟合。`predict(rows) -> np.ndarray`，形状 `(n, 2)`。
- `constant_rates(train_rows) -> tuple[float, float]`

**简单特征，都在 \(T\) 朝 \(+x\) 的坐标系里：**

- `ball_x / pitch_x`
- `ball_y / pitch_y`
- 到本方球门的距离 `/ pitch_x`，本方球门在 \(x=0\)
- 到对方球门的距离 `/ pitch_x`，对方球门在 \(x=pitch_x\)
- `possession_by_t`

`StandardScaler` 只在训练集上拟合。不要用 `class_weight="balanced"`，否则 Brier 会被故意拉偏。

- [ ] **步骤 1：测试特征**

球在 \(T\) 坐标系的 `(10, 34)`、球场 105×68。到本方球门的归一化距离约为 `10/105`，到对方球门约为 `95/105`。

- [ ] **步骤 2：测试常数**

四条样本里 `y_for` 为 `[1,0,0,0]` 时，常数第一头是 `0.25`。

- [ ] **步骤 3：实现并跑通测试**

---

### 任务 5：空间价值网络

**文件：**

- `soccervalue/value_model.py`
- 测试 `tests/test_state_value.py`

**产生：** `StateValueNet` 和 `train_state_value(train_rows, val_rows) -> tuple[StateValueNet, list[dict]]`

**网络：**

- 输入栅格调用现有 `raster(team, opponent, ball, pitch)`，通道顺序就是 \(T\)、对方、球。
- 卷积可以照 `SoccerMapSuccess` 的细尺度和粗尺度，但最后不是整张概率图。
- 对融合后的特征做全局平均，再和 `ball_x/pitch_x`、`ball_y/pitch_y`、`possession_by_t` 拼在一起，送进一个小 MLP。
- 两个 sigmoid。损失是两头二元交叉熵之和。
- 早停看验证集两头 Brier 的平均值，耐心 4，最多 20 个 epoch。保存验证最好的权重。

球坐标必须进 MLP。只有全局平均时，模型无法区分「球在禁区」和「球在中圈」。这是这个网络和成功率图的分界：这里要一个局面的两个数，不是每个落点一个数。

- [ ] **步骤 1：测试输出形状**

`StateValueNet` 对 `(2, 3, 68, 104)` 的输出形状是 `(2, 2)`，数值都在 0 和 1 之间。

- [ ] **步骤 2：测试损失用到两头**

固定第一头的标签，只改第二头的预测，损失必须变化。

- [ ] **步骤 3：实现网络和训练函数，跑通测试**

---

### 任务 6：训练入口、主结果和窗口敏感性

**文件：**

- 新建 `soccervalue/run_state_value.py`
- 复用 `soccervalue/metrics.py` 的 `binary_scores` 和 `calibration_table`

**流程：**

1. 按 `MATCH_COMPETITION` 读每场的事件和追踪。追踪文件的查找方式与 `run_pass_success.py` 相同。
2. `load_timeline` 之后先写 `outputs/state_value/label_counts.json`：每场射门数、进球数，以及训练、验证、测试里 `y_for`、`y_against` 的正例数。正例没有同时出现在三个划分里时，文件里写明「这是管线原型」。
3. `build_state_samples(..., window_s=10)`，再 `attach_state_frame`，再用训练集选出的变换做 `orient_state`。丢弃没有帧或朝向失败的样本。
4. 拟合常数、简单模型和 CNN。CNN 权重写到 `outputs/state_value/state_value_net.pt`。
5. 验证集和测试集分别记录两头的 Brier、log-loss、AUROC。校准表各写一份。
6. 用 5 秒和 20 秒重建样本并重拟合三个模型，结果放进 `metrics_w5.json` 和 `metrics_w20.json`。主结果在 `metrics_w10.json`。不要把三个窗口的损失相加。
7. 从 10 秒测试集导出 `failures_test.json`：`y_for=0` 但 \(\hat p_{\text{for}}>0.5\) 的最高几条，以及 `y_for=1` 但 \(\hat p_{\text{for}}\) 最低的几条。对方那一头同样各留几条。每条带上 `match_id`、`event_id`、两头标签、三个模型的概率，以及窗口内真实射门的球队和时间。

**阅读结果时只比较同一窗口、同一头：**

- 简单模型是否低于常数的 Brier。
- 空间模型是否低于简单模型的 Brier。
- 这个差距在验证场和测试场是否同向。
- 5 秒和 20 秒上，空间模型相对简单模型的 Brier 差距是否还是同一方向。

空间模型没有更好，不能写成「站位没有用」。先看正例数、截断丢弃率和失败样例。

- [ ] **步骤 1：用任务 2 的小时间线做一次端到端冒烟测试**，不读真实 XML。`build_state_samples` 的标签经过常数预测后，Brier 可以被 `binary_scores` 接收。
- [ ] **步骤 2：实现 `run_state_value.py`**
- [ ] **步骤 3：`python -m unittest tests.test_pass_success tests.test_state_value -v`**
- [ ] **步骤 4：`python -m soccervalue.run_state_value`**

跑完后应存在 `label_counts.json`、`metrics_w10.json`、`metrics_w5.json`、`metrics_w20.json`、`state_value_net.pt`、`failures_test.json`。

---

## 本计划故意不做的事

- 不把成功率 CNN 的最后一层改成价值。
- 不采样未发生的传球，不计算 \(Q(s,a)\)。
- 不用世界模型生成的局面来训练或早停。以后如果要打分，真实局面和模拟局面的指标分文件写。
- 不把「后继状态再往后看 10 秒」说成「从出脚时刻起的固定 10 秒」。那是两个定义，正式接世界模型之前必须分开报告。
- 不在这一轮用进球或 `xG` 当训练标签。`ShotAtGoal` 上的 `xG` 已经读进时间线，留给射门标签跑通之后的 v1。
