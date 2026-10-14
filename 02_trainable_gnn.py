"""第 2 课：用正常的 Python/PyTorch 写法训练一个小 GNN。"""

import torch


torch.manual_seed(20261014)

# 3 根杆：(0,1)、(1,2)、(1,3)
edge_index = torch.tensor(
    [[0, 1, 1],
     [1, 2, 3]],
    dtype=torch.long,
)
source_nodes = edge_index[0]
target_nodes = edge_index[1]


# Python：def 定义函数；return 一次返回三个结果。
# MATLAB 对应写法：function [node_u, edge_k, energy] = make_dataset(sample_count)
def make_dataset(sample_count):
    node_u = 2.0 * torch.rand(sample_count, 4, 1) - 1.0
    node_u[:, 0, :] = 0.0

    edge_k = 0.5 + 1.5 * torch.rand(sample_count, 3, 1)
    delta_u = node_u[:, target_nodes, :] - node_u[:, source_nodes, :]
    energy = (0.5 * edge_k * delta_u.square()).sum(dim=1)
    return node_u, edge_k, energy


# Python：class 定义类；括号里的 torch.nn.Module 是父类。
class GNN(torch.nn.Module):
    # __init__ 类似 MATLAB 类的构造函数。
    def __init__(self):
        super().__init__()
        self.message_1 = torch.nn.Linear(3, 16)
        self.message_2 = torch.nn.Linear(16, 8)
        self.update = torch.nn.Linear(9, 8)
        self.output = torch.nn.Linear(8, 1)

    # 写 model(node_u, edge_k) 时，PyTorch 会自动调用 forward。
    def forward(self, node_u, edge_k):
        message_sources = torch.cat((source_nodes, target_nodes))
        message_targets = torch.cat((target_nodes, source_nodes))
        message_edge_k = torch.cat((edge_k, edge_k), dim=1)

        message_input = torch.cat(
            (
                node_u[:, message_sources, :],
                node_u[:, message_targets, :],
                message_edge_k,
            ),
            dim=2,
        )
        messages = torch.relu(self.message_1(message_input))
        messages = torch.relu(self.message_2(messages))

        aggregate = torch.zeros(node_u.shape[0], 4, 8)
        aggregate.index_add_(1, message_targets, messages)
        degree = torch.bincount(message_targets, minlength=4).reshape(1, 4, 1)
        aggregate = aggregate / degree

        node_input = torch.cat((node_u, aggregate), dim=2)
        node_state = torch.relu(self.update(node_input))
        graph_state = node_state.mean(dim=1)
        return self.output(graph_state)


# Python：函数返回的三个结果按顺序赋给三个变量。
node_u, edge_k, target_energy = make_dataset(320)

# Python 切片和 MATLAB 很像，但 Python 从 0 开始，且末端 256 不包含在内。
train_node_u = node_u[:256]
train_edge_k = edge_k[:256]
train_target = target_energy[:256]

validation_node_u = node_u[256:]
validation_edge_k = edge_k[256:]
validation_target = target_energy[256:]

print("train_node_u.shape =", tuple(train_node_u.shape))
print("train_edge_k.shape =", tuple(train_edge_k.shape))
print("train_target.shape =", tuple(train_target.shape))


model = GNN()
loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# 每轮训练固定四步：清梯度 -> 前向 -> 反向 -> 更新参数。
for epoch in range(1, 1001):
    optimizer.zero_grad()
    prediction = model(train_node_u, train_edge_k)
    loss = loss_function(prediction, train_target)
    loss.backward()
    optimizer.step()

    if epoch == 1 or epoch % 200 == 0:
        print("epoch =", epoch, "train_mse =", float(loss.detach()))


# with 代码块结束后会自动恢复梯度设置，不需要 MATLAB 的 end。
with torch.no_grad():
    prediction = model(validation_node_u, validation_edge_k)
    validation_mae = (prediction - validation_target).abs().mean()
    residual = ((prediction - validation_target) ** 2).sum()
    total = ((validation_target - validation_target.mean()) ** 2).sum()
    validation_r_squared = 1.0 - residual / total


print("validation_mae =", float(validation_mae))
print("validation_r_squared =", float(validation_r_squared))
print("first 5 targets =\n", validation_target[:5])
print("first 5 predictions =\n", prediction[:5])
