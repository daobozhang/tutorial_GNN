"""第 1 课：不用训练，手工走完一次 GNN 消息传递。"""

# PyTorch API：整个项目只导入 torch，不使用 PyTorch Geometric。
import torch


# PyTorch API：让打印结果更容易读；它不会改变张量本身。
torch.set_printoptions(precision=4, sci_mode=False)

# PyTorch API：固定随机种子，因此每次运行都会得到相同的 Linear 初始参数。
torch.manual_seed(7)


# 自定义变量 node_features：4 行对应 4 个节点，2 列对应 2 个节点特征。
# 第 0 列是节点的 x 坐标，第 1 列是节点的标量位移 u。
node_features = torch.tensor(
    [
        [0.0, 0.0],
        [1.0, 0.2],
        [2.0, -0.1],
        [1.0, 0.4],
    ],
    dtype=torch.float32,
)

# 自定义变量 physical_edge_index：3 根真实杆件，每一列是一根杆。
# 第一行是起点编号，第二行是终点编号。
physical_edge_index = torch.tensor(
    [
        [0, 1, 1],
        [1, 2, 3],
    ],
    dtype=torch.long,
)

# 自定义变量 physical_edge_features：每根杆只有 1 个特征，即无量纲刚度。
physical_edge_features = torch.tensor(
    [
        [1.0],
        [0.8],
        [1.2],
    ],
    dtype=torch.float32,
)


print("\n========== ORIGINAL GRAPH DATA ==========")
print("node_features =\n", node_features)
print("node_features.shape =", tuple(node_features.shape))
print("physical_edge_index =\n", physical_edge_index)
print("physical_edge_index.shape =", tuple(physical_edge_index.shape))
print("physical_edge_features =\n", physical_edge_features)
print("physical_edge_features.shape =", tuple(physical_edge_features.shape))


# 一根真实杆是无向的，但消息需要沿两个方向传播。
# PyTorch API torch.cat：把原方向和反方向沿列方向拼起来。
message_edge_index = torch.cat(
    (physical_edge_index, physical_edge_index.flip(dims=(0,))),
    dim=1,
)

# 反方向消息仍对应同一根杆，所以刚度特征复制一次。
message_edge_features = torch.cat(
    (physical_edge_features, physical_edge_features),
    dim=0,
)

# Python 语法：用索引取出每条有向消息的起点和终点。
source_nodes = message_edge_index[0]
target_nodes = message_edge_index[1]


print("\n========== 3 PHYSICAL EDGES TO 6 DIRECTED MESSAGES ==========")
print("message_edge_index =\n", message_edge_index)
print("message_edge_index.shape =", tuple(message_edge_index.shape))
print("source_nodes =", source_nodes)
print("target_nodes =", target_nodes)
print("message_edge_features =\n", message_edge_features)
print("message_edge_features.shape =", tuple(message_edge_features.shape))


# PyTorch API torch.nn.Linear：定义“消息计算”和“节点更新”中的线性变换。
# 这里不训练，参数只是由固定随机种子产生的固定示例值。
message_linear = torch.nn.Linear(in_features=3, out_features=2, bias=True)
update_linear = torch.nn.Linear(in_features=4, out_features=2, bias=True)
output_linear = torch.nn.Linear(in_features=2, out_features=1, bias=True)

# PyTorch API requires_grad_：明确关闭梯度；本文件没有优化器，也不会训练。
message_linear.requires_grad_(False)
update_linear.requires_grad_(False)
output_linear.requires_grad_(False)


print("\n========== FIXED LINEAR PARAMETERS: NO TRAINING ==========")
print("message_linear.weight =\n", message_linear.weight)
print("message_linear.bias =\n", message_linear.bias)
print("update_linear.weight =\n", update_linear.weight)
print("update_linear.bias =\n", update_linear.bias)
print("output_linear.weight =\n", output_linear.weight)
print("output_linear.bias =\n", output_linear.bias)


# PyTorch 张量索引：为 6 条消息分别取出其起点节点的特征。
source_node_features = node_features[source_nodes]

# PyTorch API torch.cat：把“起点节点特征”和“边特征”拼成消息输入。
# 每条消息的输入共有 3 列：[起点 x, 起点 u, 杆件刚度]。
message_input = torch.cat((source_node_features, message_edge_features), dim=1)

# PyTorch API：Linear 做线性变换，relu 做非线性变换。
messages_before_relu = message_linear(message_input)
messages = torch.relu(messages_before_relu)


print("\n========== STEP 1: COMPUTE EDGE MESSAGES ==========")
print("source_node_features =\n", source_node_features)
print("source_node_features.shape =", tuple(source_node_features.shape))
print("message_input =\n", message_input)
print("message_input.shape =", tuple(message_input.shape))
print("messages_before_relu =\n", messages_before_relu)
print("messages_before_relu.shape =", tuple(messages_before_relu.shape))
print("messages =\n", messages)
print("messages.shape =", tuple(messages.shape))


# PyTorch API torch.zeros：先为 4 个节点各准备一个全零的消息容器。
aggregated_message_sum = torch.zeros(
    size=(4, 2),
    dtype=torch.float32,
)

# PyTorch API index_add_：按照 target_nodes，把消息加到对应目标节点。
aggregated_message_sum.index_add_(
    dim=0,
    index=target_nodes,
    source=messages,
)

# 统计每个节点收到多少条消息，随后把“求和”变成“求平均”。
incoming_message_count = torch.zeros(
    size=(4, 1),
    dtype=torch.float32,
)
incoming_message_count.index_add_(
    dim=0,
    index=target_nodes,
    source=torch.ones(size=(6, 1), dtype=torch.float32),
)
aggregated_message_mean = aggregated_message_sum / incoming_message_count.clamp_min(1.0)


print("\n========== STEP 2: AGGREGATE BY TARGET NODE ==========")
print("aggregated_message_sum =\n", aggregated_message_sum)
print("aggregated_message_sum.shape =", tuple(aggregated_message_sum.shape))
print("incoming_message_count =\n", incoming_message_count)
print("incoming_message_count.shape =", tuple(incoming_message_count.shape))
print("aggregated_message_mean =\n", aggregated_message_mean)
print("aggregated_message_mean.shape =", tuple(aggregated_message_mean.shape))


# 每个节点把“自己的旧特征”和“邻居聚合消息”拼起来，再做一次更新。
node_update_input = torch.cat((node_features, aggregated_message_mean), dim=1)
node_state_before_relu = update_linear(node_update_input)
updated_node_state = torch.relu(node_state_before_relu)


print("\n========== STEP 3: UPDATE NODE STATES ==========")
print("node_update_input =\n", node_update_input)
print("node_update_input.shape =", tuple(node_update_input.shape))
print("node_state_before_relu =\n", node_state_before_relu)
print("node_state_before_relu.shape =", tuple(node_state_before_relu.shape))
print("updated_node_state =\n", updated_node_state)
print("updated_node_state.shape =", tuple(updated_node_state.shape))


# PyTorch API mean：对 4 个节点做平均池化，得到整张图的状态。
graph_state = torch.mean(updated_node_state, dim=0, keepdim=True)

# 最后一个 Linear 把图状态映射成 1 个标量。
# 因为没有训练，这个标量只是演示数据流，不代表正确的物理预测。
untrained_prediction = output_linear(graph_state)


print("\n========== STEP 4: POOL NODES INTO ONE GRAPH ==========")
print("graph_state =\n", graph_state)
print("graph_state.shape =", tuple(graph_state.shape))
print("untrained_prediction =\n", untrained_prediction)
print("untrained_prediction.shape =", tuple(untrained_prediction.shape))
print("\nFinished: forward pass only; no loss, backward, or optimizer.")
