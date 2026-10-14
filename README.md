# tutorial_GNN

这是一个只用 PyTorch 基础功能编写的中文 GNN 入门项目。它面向熟悉 MATLAB、刚开始学习 Python 和 PyTorch 的读者。

项目不依赖原来的 GraphGPS 代码，也不使用真实科研数据。所有样本都在脚本中自动生成。

## 这个项目解决什么任务

我们使用同一张 4 节点、3 杆件的小图：

```text
0 ----- 1 ----- 2
        |
        |
        3
```

- 节点特征：节点的标量位移 `u`。
- 边特征：杆件刚度 `k`。
- 图级目标：三根杆的总能量。

目标由下面的解析公式直接生成：

```text
E = 1/2 * sum(k * (u_j - u_i)^2)
```

这个任务足够小，可以逐行检查；同时目标同时依赖节点和边，所以消息传递不能被省略。

## 文件与阅读顺序

### 1. `01_manual_message_passing.py`

先读、先运行这个文件。

它只演示一次前向传播，不训练。三个 `torch.nn.Linear` 使用固定随机参数，而且梯度被明确关闭。文件会依次打印：

1. 节点特征；
2. 3 根真实杆件的 `physical_edge_index`；
3. 由 3 根无向杆得到的 6 条有向消息；
4. 每条边的消息输入和消息输出；
5. 按目标节点聚合后的结果；
6. 更新后的节点状态；
7. 平均池化得到的图状态；
8. 未训练的图级输出。

这里最重要的区别是：**物理上仍然只有 3 根杆；6 表示消息传播方向的数量，不是 6 根杆。**

### 2. `02_trainable_gnn.py`

理解第一个文件后再读这个文件。

它会自动生成 320 个样本，其中 256 个用于训练、64 个用于验证。模型只有：

- 一次消息传递；
- 一次按目标节点聚合；
- 一次节点状态更新；
- 一次平均池化；
- 一个图级标量输出。

模型没有 Transformer、PyTorch Geometric、`torch.nn.ModuleList`、`torch.nn.Sequential`、DataLoader 或配置系统。

## 如何运行

在 PowerShell 中进入本目录：

```powershell
cd D:\Daobo\portfolio\tutorial_GNN
```

先运行不训练的例子：

```powershell
python 01_manual_message_passing.py
```

再运行可训练例子：

```powershell
python 02_trainable_gnn.py
```

项目唯一依赖是 PyTorch。如果当前 Python 环境没有 PyTorch，可运行：

```powershell
python -m pip install torch
```

## 在 VS Code 中逐行调试

1. 在 VS Code 中打开整个 `tutorial_GNN` 文件夹。
2. 先打开 `01_manual_message_passing.py`。
3. 在想暂停的代码行左侧单击，出现红点即为断点。
4. 建议第一个断点设在 `source_node_features = node_features[source_nodes]`。
5. 按 `F5`，选择当前安装了 PyTorch 的 Python 解释器。
6. 每次按 `F10` 执行一行，在 Variables 面板展开张量。
7. 重点观察张量的值以及 `.shape` 如何变化。

建议按以下位置依次设断点：

- `message_edge_index = torch.cat(...)`：3 根无向杆变成 6 条有向消息；
- `messages = torch.relu(...)`：逐边消息计算完成；
- `aggregated_message_sum.index_add_(...)`：消息按目标节点相加；
- `updated_node_state = torch.relu(...)`：节点状态更新；
- `graph_state = torch.mean(...)`：节点状态池化成图状态；
- `train_loss.backward()`：PyTorch 计算梯度；
- `optimizer.step()`：参数真正更新。

## Python、PyTorch 与自定义名称如何区分

| 类型 | 本项目中的例子 | 含义 |
| --- | --- | --- |
| Python 语法 | `class`、`def`、`for`、`if`、赋值号 `=` | Python 语言本身规定的写法 |
| PyTorch API | `torch.tensor`、`torch.cat`、`torch.nn.Linear`、`torch.relu`、`loss.backward` | PyTorch 提供的功能 |
| 自定义变量 | `node_features`、`message_edge_index`、`aggregated_message_sum` | 为了说明含义，由本项目自己命名 |
| 自定义类 | `OneMessagePassingGNN` | 本项目自己定义的模型名称 |

看到不认识的名称时，可以先检查它是否以 `torch.` 开头：

- 以 `torch.` 开头，通常是 PyTorch API；
- 不以 `torch.` 开头，通常是 Python 语法或本项目自己定义的名称。

## 哪些东西固定，哪些东西会学习

| 内容 | `01_manual_message_passing.py` | `02_trainable_gnn.py` |
| --- | --- | --- |
| 4 节点、3 杆拓扑 | 固定 | 固定 |
| 节点特征和边特征 | 手工固定一个样本 | 自动随机生成多个样本 |
| Linear 参数 | 固定，不训练 | 由训练更新 |
| 解析目标能量 | 不使用 | 固定公式生成 |
| `edge_index` | 不学习 | 不学习 |
| 消息传递顺序 | 代码规定 | 代码规定 |

`edge_index` 不是模型参数。训练改变的是各个 `torch.nn.Linear` 内部的 `weight` 和 `bias`。

## 推荐的学习方式

不要一开始就修改模型。先完成下面三个检查：

1. 手工说出 `message_edge_index` 六列各代表哪一条消息；
2. 在断点处确认节点 1 收到三条消息，而节点 0、2、3 各收到一条消息；
3. 确认只有执行 `optimizer.step()` 后，Linear 参数才发生变化。

完成这三步后，再尝试修改节点数、杆件数或增加第二次消息传递。
