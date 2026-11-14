# VAEP

Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions Speak Louder than Goals: Valuing Player Actions in Soccer. In *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (KDD ’19), pp. 1851–1861. https://doi.org/10.1145/3292500.3330758

## 链接

- 正式出版：https://doi.org/10.1145/3292500.3330758
- 预印本：https://arxiv.org/abs/1802.07127
- 预印本 HTML：https://arxiv.org/html/1802.07127
- 预印本 PDF：https://arxiv.org/pdf/1802.07127
- 实现：https://github.com/ML-KULeuven/socceraction
- VAEP 文档：https://socceraction.readthedocs.io/en/stable/documentation/valuing_actions/vaep.html
- SPADL 文档：https://socceraction.readthedocs.io/en/latest/documentation/spadl/spadl.html

## 它解决什么

论文同时给出两件东西。SPADL（Soccer Player Action Description Language）把不同厂商的事件流收成统一的动作记录，字段包括起止时间、起终点、球员、球队、动作类型和结果。VAEP 在这条动作序列上评价每个已发生动作：分别预测该动作之后较短未来内的进球概率和失球概率，动作价值取两者变化之差。一个正值表示该动作提高了本方净进球期望。

socceraction 中的默认设定是：用最近三个动作描述当前状态，预测未来十个动作内的进球与失球概率。这是事件序列上的价值框架，状态里没有全体球员的实时站位。

## 在本题中的位置

这条线保留为基线：已有事件数据先转成 SPADL，再复现事件型 VAEP。它评价的是已经发生、结果已知的动作。候选传球在出脚前没有真实终点结果，不能把未来结果填进 SPADL 再交给 VAEP，否则会用到当时尚不存在的信息。
