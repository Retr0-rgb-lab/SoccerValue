# SoccerMap

Fernández, J., & Bornn, L. (2021). SoccerMap: A Deep Learning Architecture for Visually-Interpretable Analysis in Soccer. In *Machine Learning and Knowledge Discovery in Databases. Applied Data Science and Demo Track* (ECML PKDD 2020, Ghent), Lecture Notes in Computer Science, pp. 491–506. https://doi.org/10.1007/978-3-030-67670-4_30

## 链接

- 正式出版：https://doi.org/10.1007/978-3-030-67670-4_30
- 预印本：https://arxiv.org/abs/2010.10202
- 预印本 PDF：https://arxiv.org/pdf/2010.10202

## 它解决什么

SoccerMap 是一个全卷积网络。输入是由追踪数据做成的低层空间通道，输出是全场每个位置上的预测。网络在原始尺度、二分之一尺度和四分之一尺度上提取特征，再把各尺度的预测融合。细尺度保留落点附近的细节，粗尺度覆盖更大范围的空间关系。训练时每个样本只有一个落点标签，网络仍要给出整张预测图。

论文用同一架构处理三张不同的图：传球成功概率、传球选择概率、传球期望价值。三者是不同的预测目标。

## 在本题中的位置

SoccerMap 用在感知之后。球员和球的位置先成为空间状态，再由这个网络生成预测图。它本身不做球员检测，也不是一张固定分区的评分表。热力图可以直接作为预测结果来看，选择概率、成功概率和预期价值要分开画。用于推荐的主图是预期价值，成功概率用来解释风险。
