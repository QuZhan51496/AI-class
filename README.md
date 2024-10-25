# AI-class

> 浙江大学《人工智能算法与系统》课程作业

- 成人死亡率预测：把线性预测换成随机森林即可
- 金融异常检测：把Linear换成GCNConv即可
- 机器人自动走迷宫：唯一要注意的是DQN一定要走到终点才算通过。给的TorchRobot类训练不稳定，需要自己写一个函数训练模型直到能走到终点。参考：https://github.com/QikaiXu/Robot-Maze-Solving