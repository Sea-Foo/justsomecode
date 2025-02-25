import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd

def prepare_data(data):
    """
    处理数据集，将每10个样本合成一个feature（维度70），输出为第11个样本（维度7）

    参数：
    data: numpy 数组，形状 (2000, 7)

    返回：
    X: numpy 数组，特征数据，形状 (n_samples, 70)
    y: numpy 数组，目标数据，形状 (n_samples, 7)
    """
    X, y = [], []
    for i in range(len(data) - 20):
        X.append(data[i:i+20].flatten())  # 展平成 70 维
        y.append(data[i+20])  # 第11个样本作为输出
    return np.array(X), np.array(y)

# 生成示例数据 或 读取数据 -  使用您的数据读取方式
all_data = pd.read_excel('features.xlsx').to_numpy()

# 划分训练集（前90%）和测试集（后10%）
train_size = int(0.9 * len(all_data))
train_data, test_data = all_data[:train_size], all_data[train_size:]

# 处理训练集和测试集
X_train, y_train = prepare_data(train_data)
X_test, y_test = prepare_data(test_data)

# 输出数据形状
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

print("训练集历史数据 (最近10期 示例):") # 仅打印前10期作为示例
for i, data in enumerate(X_train[:10]):
    print(f"第{i+1}期: 红球 {data[:6]}, 蓝球 {data[6]}")

# 2. 数据预处理 (为LSTM准备序列数据) - 修改为适应您prepare_data的输出
def prepare_lstm_data_for_eval(X_train, y_train, X_test, y_test, sequence_length=20): # 函数修改为接收训练集和测试集
    # 转换为Tensor 并进行标准化 (这里简单 Min-Max 标准化到 0-1 范围)
    normalized_input_sequences_train = []
    normalized_output_sequences_train = []
    normalized_input_sequences_test = []
    normalized_output_sequences_test = []

    #  处理 训练集
    for input_seq in X_train:
        #  由于 prepare_data 展平了10期数据，这里需要reshape回 (sequence_length, feature_dim)
        input_seq_reshaped = input_seq.reshape(sequence_length, 7)
        normalized_seq = []
        for numbers in input_seq_reshaped: # 遍历每期号码 (在序列中的)
            normalized_numbers = [(x - 1) / 32 for x in numbers[:6]] + [(numbers[6] - 1) / 16] # 红球和蓝球分别标准化
            normalized_seq.append(normalized_numbers)
        normalized_input_sequences_train.append(torch.tensor(normalized_seq).float())

    for output_seq in y_train:
        normalized_output = [(x - 1) / 32 for x in output_seq[:6]] + [(output_seq[6] - 1) / 16]
        normalized_output_sequences_train.append(torch.tensor(normalized_output).float())


    # 处理 测试集 - 同样需要 reshape
    for input_seq in X_test:
        input_seq_reshaped = input_seq.reshape(sequence_length, 7)
        normalized_seq = []
        for numbers in input_seq_reshaped:
            normalized_numbers = [(x - 1) / 32 for x in numbers[:6]] + [(numbers[6] - 1) / 16]
            normalized_seq.append(normalized_numbers)
        normalized_input_sequences_test.append(torch.tensor(normalized_seq).float())

    for output_seq in y_test:
        normalized_output = [(x - 1) / 32 for x in output_seq[:6]] + [(output_seq[6] - 1) / 16]
        normalized_output_sequences_test.append(torch.tensor(normalized_output).float())


    # 为预测准备输入数据 -  使用 训练集 的 **最后** 10期 数据的展平数据，同样需要 reshape
    input_sequence_for_prediction_np = X_train[-1:] # 取训练集最后一条展平数据 (1x70)
    input_sequence_for_prediction_reshaped = input_sequence_for_prediction_np.reshape(sequence_length, 7) # reshape (10x7)
    normalized_input_prediction = []
    for numbers in input_sequence_for_prediction_reshaped:
        normalized_numbers = [(x - 1) / 32 for x in numbers[:6]] + [(numbers[6] - 1) / 16]
        normalized_input_prediction.append(normalized_numbers)
    input_tensor_for_prediction = torch.tensor(normalized_input_prediction).float().unsqueeze(0) # 增加batch维度


    return normalized_input_sequences_train, normalized_output_sequences_train, normalized_input_sequences_test, normalized_output_sequences_test, input_tensor_for_prediction


sequence_length = 20 #  !!!  重要：这里sequence_length 必须和 prepare_data 中的 10 保持一致，因为您的特征就是基于 10 期数据生成的
input_sequences, output_sequences, normalized_input_sequences_test, normalized_output_sequences_test, input_tensor_for_prediction = prepare_lstm_data_for_eval(X_train, y_train, X_test, y_test, sequence_length) #  修改函数调用，传入 X_train, y_train, X_test, y_test

print(f"训练集样本数: {len(input_sequences)}")
print(f"测试集样本数: {len(normalized_input_sequences_test)}") # 实际用于评估的测试集样本数


# 3. 构建 LSTM 模型
class LotteryPredictorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LotteryPredictorLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class LotteryPredictorGRU(nn.Module): #  类名保持 LotteryPredictorLSTM, 虽然内部用的是 GRU，为了代码一致性，类名可以不改
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LotteryPredictorGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.GRU(input_size, hidden_size, num_layers, batch_first=True) # 使用 nn.GRU 替换 nn.LSTM
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化 hidden state (GRU 只需要 hidden state, 不需要 cell state)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 GRU (GRU 只需要 hidden state h0)
        out, _ = self.lstm(x, h0)  # 修改这里，只传入 h0

        # 取序列最后一个时间步的输出
        out = out[:, -1, :]

        # 解码到输出维度
        out = self.fc(out)
        return out

input_size = 7 # 输入特征数 (每期彩票号码数)
hidden_size = 128
num_layers = 2
output_size = 7

# model_lstm = LotteryPredictorLSTM(input_size, hidden_size, num_layers, output_size)
model_lstm = LotteryPredictorGRU(input_size, hidden_size, num_layers, output_size)

criterion = nn.MSELoss()
optimizer = optim.Adam(model_lstm.parameters(), lr=0.001)

epochs = 1000
batch_size = 64

patience = 300 #  耐心值：验证集loss连续多少轮不下降就早停
best_val_loss = float('inf') #  最佳验证集loss，初始化为正无穷大
epochs_no_improve = 0 #  记录验证集loss连续不下降的轮数
best_model_state = None # 保存最佳模型参数

# 4. 模型训练 (与之前代码相同，使用训练集训练)
for epoch in range(epochs):
    total_loss = 0
    for i in range(0, len(input_sequences), batch_size): # 使用 训练集  !!!
        inputs = torch.stack(input_sequences[i:i+batch_size]) # 使用 标准化后的训练集输入  !!!
        targets = torch.stack(output_sequences[i:i+batch_size]) # 使用 标准化后的训练集输出  !!!

        optimizer.zero_grad()
        outputs = model_lstm(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    ###  验证集评估  ###
    model_lstm.eval() # 设置为评估模式，不计算梯度
    val_loss = 0
    with torch.no_grad(): #  不计算梯度
        for i in range(0, len(normalized_input_sequences_test), batch_size): #  遍历验证集
            inputs_val = torch.stack(normalized_input_sequences_test[i:i+batch_size]) #  验证集输入
            targets_val = torch.stack(normalized_output_sequences_test[i:i+batch_size]) # 验证集目标输出
            outputs_val = model_lstm(inputs_val) #  验证集前向传播
            val_loss += criterion(outputs_val, targets_val).item() #  计算验证集loss
    avg_val_loss = val_loss / (len(normalized_input_sequences_test) // batch_size + 1)

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], 平均Loss: {total_loss / (len(input_sequences) // batch_size + 1):.4f}')
        print(f'验证集Loss: {avg_val_loss:.4f}')

    ###  早停逻辑  ###
    if avg_val_loss < best_val_loss: #  验证集loss下降
        best_val_loss = avg_val_loss #  更新最佳验证集loss
        epochs_no_improve = 0 # 重置计数器
        best_model_state = model_lstm.state_dict() # 保存当前模型参数
    else: # 验证集loss没有下降
        epochs_no_improve += 1 # 计数器增加
        if epochs_no_improve >= patience: # 达到耐心值，触发早停
            print(f"早停触发! Epoch [{epoch+1}/{epochs}], 验证Loss连续 {patience} 轮未下降. 停止训练.")
            model_lstm.load_state_dict(best_model_state) # 加载最佳模型参数
            break # 停止训练循环

# 保存模型
torch.save(model_lstm.state_dict(), 'lottery_predictor_lstm.pth')

###  测试集评估部分  ###

# 5.  在测试集上进行预测并计算指标
model_lstm.eval() # 评估模式
total_red_ball_matches = 0
blue_ball_matches = 0
num_test_samples = len(normalized_input_sequences_test) #  实际用于评估的测试样本数
all_ball_matches = 0

with torch.no_grad():
    for i in range(num_test_samples): # 遍历测试集样本
        input_tensor_test = normalized_input_sequences_test[i].unsqueeze(0) # 取出测试集输入
        target_output_test = normalized_output_sequences_test[i] # 取出测试集目标输出 (实际号码)
        actual_numbers_test = y_test[i] #  !!!  测试集的真实号码，从 y_test 中获取

        predicted_output_normalized = model_lstm(input_tensor_test) # 模型预测 (标准化输出)
        predicted_numbers_normalized = predicted_output_normalized.squeeze().tolist()

        # 反标准化预测结果
        predicted_red_balls = [int(round(predicted_numbers_normalized[i] * 33 + 1)) for i in range(6)] # 反标准化 红球
        predicted_blue_ball = int(round(predicted_numbers_normalized[6] * 16 + 1)) # 反标准化 蓝球


        # 确保范围 和 去重 (与预测代码相同)
        predicted_red_balls = [int(round(max(1, min(x, 33)))) for x in predicted_red_balls]
        predicted_blue_ball = int(round(max(1, min(predicted_blue_ball, 16))))
        predicted_red_balls_unique = []
        for ball in sorted(predicted_red_balls, reverse=True):
            if ball not in predicted_red_balls_unique and 1 <= ball <= 33 and len(predicted_red_balls_unique) < 6:
                predicted_red_balls_unique.append(ball)
        if len(predicted_red_balls_unique) < 6:
            for ball in predicted_red_balls:
                if len(predicted_red_balls_unique) >= 6:
                    break
                if ball not in predicted_red_balls_unique and 1 <= ball <= 33:
                     predicted_red_balls_unique.append(ball)
        while len(predicted_red_balls_unique) < 6:
            new_red = random.randint(1, 33)
            if new_red not in predicted_red_balls_unique:
                predicted_red_balls_unique.append(new_red)
        predicted_red_balls = sorted(predicted_red_balls_unique)[:6]


        actual_red_balls = actual_numbers_test[:6] #  实际红球，从 actual_numbers_test 获取
        actual_blue_ball = actual_numbers_test[6] # 实际蓝球，从 actual_numbers_test 获取

        # 计算红球命中个数 (交集大小)
        red_ball_matches = len(set(predicted_red_balls).intersection(set(actual_red_balls)))
        total_red_ball_matches += red_ball_matches

        # 检查蓝球是否命中
        if predicted_blue_ball == actual_blue_ball:
            blue_ball_matches += 1

        # 计算所有号码命中个数
        all_ball_matches += red_ball_matches == 6 + (1 if predicted_blue_ball == actual_blue_ball else 0)


###  测试集评估结果输出  ###
average_red_ball_matches = total_red_ball_matches / num_test_samples
blue_ball_accuracy = blue_ball_matches / num_test_samples * 100 # 百分比
all_ball_accuracy = all_ball_matches / num_test_samples * 100 # 百分比


print("\n-------- 测试集评估结果 --------")
print(f"测试集样本数: {num_test_samples}")
print(f"平均每期红球命中个数: {average_red_ball_matches:.2f} 个")
print(f"蓝球命中率: {blue_ball_accuracy:.2f}%")
print(f"所有号码命中率: {all_ball_accuracy:.2f}%")
print("------------------------------")


###  单期预测 (仍然保留单期预测功能，使用训练集最后 sequence_length 期作为输入) ###
model_lstm.eval() # 评估模式
with torch.no_grad():
    predicted_output = model_lstm(input_tensor_for_prediction) # 使用  训练集  最后数据预测

predicted_numbers_normalized = predicted_output.squeeze().tolist()
predicted_red_balls = [int(round(predicted_numbers_normalized[i] * 33 + 1)) for i in range(6)]
predicted_blue_ball = int(round(predicted_numbers_normalized[6] * 16 + 1))
predicted_red_balls = [int(round(max(1, min(x, 33)))) for x in predicted_red_balls]
predicted_blue_ball = int(round(max(1, min(predicted_blue_ball, 16))))
predicted_red_balls_unique = []
for ball in sorted(predicted_red_balls, reverse=True):
    if ball not in predicted_red_balls_unique and 1 <= ball <= 33 and len(predicted_red_balls_unique) < 6:
        predicted_red_balls_unique.append(ball)
if len(predicted_red_balls_unique) < 6:
    for ball in predicted_red_balls:
        if len(predicted_red_balls_unique) >= 6:
            break
        if ball not in predicted_red_balls_unique and 1 <= ball <= 33:
             predicted_red_balls_unique.append(ball)
    while len(predicted_red_balls_unique) < 6:
        new_red = random.randint(1, 33)
        if new_red not in predicted_red_balls_unique:
            predicted_red_balls_unique.append(new_red)
predicted_red_balls = sorted(predicted_red_balls_unique)[:6]


print("\n--------  单期预测结果 (使用训练集最后数据) --------")
print("预测红球:", predicted_red_balls)
print("预测蓝球:", predicted_blue_ball)
print("----------------------------------------------------")