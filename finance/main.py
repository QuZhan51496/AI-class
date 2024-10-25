from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric.transforms as T

import numpy as np
from torch_geometric.data import Data
import os

#设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


path='./datasets/632d74d4e2843a53167ee9a1-momodel/' #数据保存路径
save_dir='./results/' #模型保存路径
dataset_name='DGraph'
dataset = DGraphFin(root=path, name=dataset_name, transform=T.ToSparseTensor())

nlabels = dataset.num_classes
if dataset_name in ['DGraph']:
    nlabels = 2    #本实验中仅需预测类0和类1

data = dataset[0]
data.adj_t = data.adj_t.to_symmetric() #将有向图转化为无向图


if dataset_name in ['DGraph']:
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
if data.y.dim() == 2:
    data.y = data.y.squeeze(1)

split_idx = {'train': data.train_mask, 'valid': data.valid_mask, 'test': data.test_mask}  #划分训练集，验证集

train_idx = split_idx['train']
result_dir = prepare_folder(dataset_name,'mlp')
    
from torch_geometric.nn import GCNConv, BatchNorm

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=-1)


epochs = 200
log_steps = 10 # log记录周期

gcn_parameters = {
    'lr': 0.01,
    'num_layers': 3,  # 图卷积层数
    'hidden_channels': 128,
    'dropout': 0.5,
    'weight_decay': 5e-4
}


para_dict = gcn_parameters
model_para = gcn_parameters.copy()
model_para.pop('lr')
model_para.pop('weight_decay')
model = GCN(in_channels=data.x.size(-1), out_channels=nlabels, **model_para).to(device)

## 生成 main.py 时请勾选此 cell
from utils import DGraphFin
from utils.evaluator import Evaluator
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import Data
import numpy as np
import os

# 这里可以加载你的模型
model.load_state_dict(torch.load('./results/model.pt', map_location=torch.device('cpu')))
out = np.load("./results/out.npy")

def predict(data,node_id):
    """
    加载模型和模型预测
    :param node_id: int, 需要进行预测节点的下标
    :return: tensor, 类0以及类1的概率, torch.size[1,2]
    """
    # -------------------------- 实现模型预测部分的代码 ---------------------------
    # with torch.no_grad():
        # model.eval()
        # row, col, _ = data.adj_t.coo()
        # edge_index = torch.stack([row, col], dim=0)
        # out = model(data.x, edge_index)
        # y_pred = out.exp()  # (N,num_classes)
        # y_pred = out[node_id]
    y_pred = out[node_id]

    return y_pred
