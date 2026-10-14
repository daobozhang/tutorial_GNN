"""第 2 课：使用自生成数据训练一个只有一层消息传递的 GNN。"""

# PyTorch API：整个文件仍然只导入 torch。
import torch


# 固定随机种子，使数据、初始参数和训练结果都可重复。
torch.manual_seed(20261014)
torch.set_printoptions(precision=4, sci_mode=False)


# 4 个节点和 3 根无向杆的固定拓扑。
physical_edge_index = torch.tensor(
    [
        [0, 1, 1],
        [1, 2, 3],
    ],
    dtype=torch.long,
)

# Python 自定义变量：一共生成 320 个小样本。
sample_count = 320

# PyTorch API torch.rand：为每个样本生成 4 个节点位移，范围是 [-1, 1]。
node_displacements = -1.0 + 2.0 * torch.rand(
    size=(sample_count, 4, 1),
    dtype=torch.float32,
)

# 把第 0 个节点固定为零位移，使例子更像一个简单杆系。
node_displacements[:, 0, 0] = 0.0

# 为每个样本的 3 根杆生成刚度，范围是 [0.5, 2.0]。
physical_edge_stiffness = 0.5 + 1.5 * torch.rand(
    size=(sample_count, 3, 1),
    dtype=torch.float32,
)

# 用解析公式生成监督目标：E = 1/2 * sum(k * (u_j - u_i)^2)。
# 这不是有限元求解，只是一个可验证、无需下载数据的通用图回归任务。
physical_source_nodes = physical_edge_index[0]
physical_target_nodes = physical_edge_index[1]
source_displacements = node_displacements[:, physical_source_nodes, :]
target_displacements = node_displacements[:, physical_target_nodes, :]
edge_extensions = target_displacements - source_displacements
target_energy = torch.sum(
    0.5 * physical_edge_stiffness * torch.square(edge_extensions),
    dim=1,
)


# 前 256 个样本用于训练，后 64 个样本只用于验证。
train_node_features = node_displacements[:256]
train_edge_features = physical_edge_stiffness[:256]
train_target_energy = target_energy[:256]
validation_node_features = node_displacements[256:]
validation_edge_features = physical_edge_stiffness[256:]
validation_target_energy = target_energy[256:]


print("\n========== SELF-GENERATED DATA ==========")
print("node_displacements.shape =", tuple(node_displacements.shape))
print("physical_edge_stiffness.shape =", tuple(physical_edge_stiffness.shape))
print("target_energy.shape =", tuple(target_energy.shape))
print("sample 0 node displacements =\n", node_displacements[0])
print("sample 0 edge stiffness =\n", physical_edge_stiffness[0])
print("sample 0 edge extensions =\n", edge_extensions[0])
print("sample 0 target energy =\n", target_energy[0])


# Python 语法 class：定义一个可训练模型。
# PyTorch API torch.nn.Module：让 PyTorch 能自动收集所有可训练参数。
class OneMessagePassingGNN(torch.nn.Module):
    """只有一次消息传递、平均池化和一个标量输出的最小 GNN。"""

    # Python 语法 __init__：创建模型组件，只在实例化模型时执行一次。
    def __init__(self):
        # PyTorch API：初始化父类 torch.nn.Module。
        super().__init__()

        # 每条消息输入 3 个数：[起点位移, 终点位移, 杆件刚度]。
        self.message_linear_1 = torch.nn.Linear(
            in_features=3,
            out_features=16,
            bias=True,
        )
        self.message_linear_2 = torch.nn.Linear(
            in_features=16,
            out_features=8,
            bias=True,
        )

        # 节点更新输入 9 个数：[自己的位移, 8 个聚合消息分量]。
        self.update_linear = torch.nn.Linear(
            in_features=9,
            out_features=8,
            bias=True,
        )

        # 图级输出把 8 维图状态变成 1 个预测能量。
        self.output_linear = torch.nn.Linear(
            in_features=8,
            out_features=1,
            bias=True,
        )

    # Python 语法 forward：明确写出模型从输入到输出的执行顺序。
    # 当代码写 model(...) 时，PyTorch 会自动调用这里。
    def forward(self, node_features, edge_index, edge_features):
        # 3 根无向杆改写为 6 条有向消息。
        message_edge_index = torch.cat(
            (edge_index, edge_index.flip(dims=(0,))),
            dim=1,
        )
        message_source_nodes = message_edge_index[0]
        message_target_nodes = message_edge_index[1]

        # 同一根杆正反两个方向使用相同的刚度。
        message_edge_features = torch.cat(
            (edge_features, edge_features),
            dim=1,
        )

        # 为每条消息取出起点位移、终点位移和杆件刚度。
        message_source_features = node_features[:, message_source_nodes, :]
        message_target_features = node_features[:, message_target_nodes, :]
        message_input = torch.cat(
            (
                message_source_features,
                message_target_features,
                message_edge_features,
            ),
            dim=2,
        )

        # 两个显式 Linear，中间使用 relu；没有 Sequential。
        message_hidden = torch.relu(self.message_linear_1(message_input))
        messages = torch.relu(self.message_linear_2(message_hidden))

        # 为每个样本、每个节点创建一个 8 维全零消息容器。
        aggregated_message_sum = torch.zeros(
            size=(node_features.shape[0], 4, 8),
            dtype=node_features.dtype,
            device=node_features.device,
        )

        # Python for 循环：6 条有向消息逐条加到目标节点。
        for message_number in range(message_target_nodes.shape[0]):
            target_node = int(message_target_nodes[message_number].item())
            aggregated_message_sum[:, target_node, :] = (
                aggregated_message_sum[:, target_node, :]
                + messages[:, message_number, :]
            )

        # 这张固定小图的入度为 [1, 3, 1, 1]，用于求平均消息。
        incoming_message_count = torch.tensor(
            [1.0, 3.0, 1.0, 1.0],
            dtype=node_features.dtype,
            device=node_features.device,
        ).reshape(1, 4, 1)
        aggregated_message_mean = aggregated_message_sum / incoming_message_count

        # 节点更新：自己的旧位移与邻居平均消息一起进入 Linear。
        node_update_input = torch.cat(
            (node_features, aggregated_message_mean),
            dim=2,
        )
        updated_node_state = torch.relu(self.update_linear(node_update_input))

        # 平均池化：4 个节点状态变成一个图状态。
        graph_state = torch.mean(updated_node_state, dim=1)

        # 图状态变成一个标量预测。
        predicted_energy = self.output_linear(graph_state)
        return predicted_energy


# 创建模型。此时所有 Linear 参数都是随机初始值，尚未学习。
model = OneMessagePassingGNN()

# PyTorch API：Adam 负责根据梯度更新 model.parameters() 中的参数。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# PyTorch API：均方误差衡量预测能量与解析目标能量之间的差别。
loss_function = torch.nn.MSELoss()


print("\n========== MODEL PARAMETERS ==========")
for parameter_name, parameter_value in model.named_parameters():
    print(parameter_name, tuple(parameter_value.shape), "requires_grad =", parameter_value.requires_grad)


print("\n========== TRAINING ==========")

# Python for 循环：完整数据很小，所以每个 epoch 直接使用全部 256 个训练样本。
for epoch in range(1, 1001):
    # 清除上一个 epoch 留下的梯度。
    optimizer.zero_grad()

    # 前向传播：model(...) 会自动进入 forward(...)。
    predicted_train_energy = model(
        train_node_features,
        physical_edge_index,
        train_edge_features,
    )

    # 计算一个标量 loss。
    train_loss = loss_function(predicted_train_energy, train_target_energy)

    # 反向传播：为所有可训练参数计算梯度。
    train_loss.backward()

    # 参数更新：这是参数真正发生变化的地方。
    optimizer.step()

    # 每 100 个 epoch 打印一次；第 1 个 epoch 也打印。
    if epoch == 1 or epoch % 100 == 0:
        # 验证时关闭梯度，因为这里只检查结果，不更新参数。
        with torch.no_grad():
            predicted_validation_energy = model(
                validation_node_features,
                physical_edge_index,
                validation_edge_features,
            )
            validation_loss = loss_function(
                predicted_validation_energy,
                validation_target_energy,
            )
        print(
            "epoch =",
            epoch,
            "train_mse =",
            float(train_loss.detach()),
            "validation_mse =",
            float(validation_loss),
        )


# 训练完成后，再对从未参与参数更新的 64 个验证样本做最终评价。
with torch.no_grad():
    predicted_validation_energy = model(
        validation_node_features,
        physical_edge_index,
        validation_edge_features,
    )
    validation_absolute_error = torch.mean(
        torch.abs(predicted_validation_energy - validation_target_energy)
    )
    residual_sum_of_squares = torch.sum(
        torch.square(predicted_validation_energy - validation_target_energy)
    )
    total_sum_of_squares = torch.sum(
        torch.square(validation_target_energy - torch.mean(validation_target_energy))
    )
    validation_r_squared = 1.0 - residual_sum_of_squares / total_sum_of_squares


print("\n========== FINAL VALIDATION ==========")
print("validation_mae =", float(validation_absolute_error))
print("validation_r_squared =", float(validation_r_squared))
print("first 5 validation targets =\n", validation_target_energy[:5])
print("first 5 validation predictions =\n", predicted_validation_energy[:5])
