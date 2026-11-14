# un-xPass

Robberechts, P., Van Roy, M., & Davis, J. (2023). un-xPass: Measuring Soccer Player’s Creativity. In *Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining* (KDD ’23), pp. 4768–4777. https://doi.org/10.1145/3580305.3599924

更早的会议版本见于 2022 StatsBomb Conference，Lirias 记录为 https://lirias.kuleuven.be/3838193 。

## 链接

- 正式出版：https://doi.org/10.1145/3580305.3599924
- Lirias 文件：https://lirias.kuleuven.be/retrieve/715497
- Lirias 记录：https://lirias.kuleuven.be/3838193
- 实现：https://github.com/ML-KULeuven/un-xPass

## 它解决什么

论文要衡量的是传球是否有创造力：既要比大多数球员在相似局面下会选的传球更不寻常，又要比那个常见选择更有价值。Creative Decision Rating 因此同时看两件事，选择的原创性，以及这个选择对进球机会的提升。

底层是三个模型，都在 StatsBomb 360 的球员站位上估计：

- 每个可能传球终点被选中的概率
- 每个传球选项的长期收益
- 每个传球选项的成功概率

每个模型都提供 SoccerMap 式的卷积网络，以及基于手工特征的梯度提升树。作者在 2021/22 英超赛季上检查这个指标，并发布了对应的 Python 包。

## 在本题中的位置

un-xPass 直接处理“实际选择”和“其他可能选择”的比较，适合作为候选传球排序的参照。它评估的是一次已观察传球相对于其他终点有多不寻常、多有价值。本题第一版要回答的是持球者各候选传球的预期价值；创造力评分是后续问题，不能代替成功率、成功后价值和失败后价值这三个估计。比较未执行动作还涉及反事实，一张看起来合理的热力图本身不是验证。
