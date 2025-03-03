import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# 资源类
class Resource:
    def __init__(self, name, quantity):
        self.name = name
        self.quantity = quantity

    def add(self, amount):
        self.quantity = max(0, self.quantity + amount)

    def subtract(self, amount):
        if self.quantity >= amount:
            self.quantity -= amount
            return True
        return False

# 工具类
class Tool:
    def __init__(self, tool_type, durability):
        self.tool_type = tool_type
        self.durability = durability

    def use(self):
        self.durability -= 1
        return self.durability > 0

# 居民类
class Citizen:
    # 修改Citizen类的初始化方法，添加默认职业
    def __init__(self, id, gender, age, wealth, food):
        self.id = id
        self.gender = gender
        self.age = age
        self.wealth = wealth * 1.5  # 增加初始财富
        self.food = food * 1.5      # 增加初始食物
        self.tool = None
        self.alive = True
        # 为新居民分配默认职业，增加农民比例
        self.profession = random.choice(['farmer', 'farmer', 'farmer', 'lumberjack', 'miner', 'blacksmith'])
        self.action_space = ['work', 'buy_food', 'reproduce', 'change_profession', 'idle']

    def get_state(self, city):
        return np.array([
            self.age,
            self.wealth,
            self.food,
            city.resources['food'].quantity,
            city.market.food_price,
            len([c for c in city.citizens if c.alive]),
            self.tool.durability if self.tool else 0
        ], dtype=np.float32)

    # 居民类中的select_action方法修改
    def select_action(self, model, epsilon, city):  # 添加city参数
        if random.random() < epsilon:
            return random.choice(range(len(self.action_space)))
        
        with torch.no_grad():
            state = torch.FloatTensor(self.get_state(city)).unsqueeze(0)
            q_values = model(state)
            return q_values.argmax().item()

    # 添加工具购买方法
    def buy_tool(self, city):
        if self.wealth < 10:  # 工具价格
            return False
        
        desired_tool = None
        if self.profession == 'lumberjack' and city.market.tools['axe']:
            desired_tool = 'axe'
        elif self.profession == 'miner' and city.market.tools['pickaxe']:
            desired_tool = 'pickaxe'
        
        if desired_tool and city.market.tools[desired_tool]:
            self.tool = city.market.tools[desired_tool].pop()
            self.wealth -= 10
            return True
        return False

    # 添加繁衍方法
    # 修改繁衍方法，降低繁衍门槛
    # 修改繁衍方法，进一步降低门槛并增加新生儿初始资源
    def reproduce(self, city):
        # 进一步降低繁衍的食物要求
        if self.gender == 'male' or self.age < 18 or self.age > 50 or self.food < 2:
            return False
        
        # 寻找合适的伴侣
        partners = [c for c in city.citizens if c.alive and c.gender == 'male' 
                    and 18 <= c.age <= 60 and c.food >= 2]
        
        if not partners:
            return False
        
        partner = random.choice(partners)
        
        # 减少消耗的资源
        self.food -= 1
        partner.food -= 1
        
        # 创建新居民，增加初始资源
        child_id = max([c.id for c in city.citizens]) + 1 if city.citizens else 0
        child_gender = random.choice(['male', 'female'])
        child = Citizen(
            id=child_id,
            gender=child_gender,
            age=0,  # 新生儿
            wealth=10,  # 给予一些初始财富
            food=10  # 增加初始食物
        )
        city.add_citizen(child)
        return True

    # 修改Citizen类的work方法，增加工作收益
    # 修改work方法，大幅增加工作收益
    def work(self, city):
        if self.profession == 'lumberjack':
            efficiency = 1
            if self.tool and self.tool.tool_type == 'axe':
                efficiency += 1
                self.tool.use()
            city.resources['wood'].add(efficiency)
            self.wealth += efficiency * 2.0  # 进一步增加收益
            
        elif self.profession == 'miner':
            efficiency = 1
            if self.tool and self.tool.tool_type == 'pickaxe':
                efficiency += 1
                self.tool.use()
            city.resources['stone'].add(efficiency)
            self.wealth += efficiency * 2.0  # 进一步增加收益
            
        elif self.profession == 'farmer':
            efficiency = 1
            city.resources['food'].add(efficiency * 5)  # 进一步增加食物产出
            self.wealth += efficiency * 2.0  # 进一步增加收益
            
        elif self.profession == 'blacksmith':
            if city.resources['wood'].subtract(1) and city.resources['stone'].subtract(2):
                tool_type = random.choice(['axe', 'pickaxe'])
                new_tool = Tool(tool_type, 10)  # 创建新工具
                city.market.add_tool(new_tool)
                self.wealth += 5.0  # 进一步增加收益

# 市场类
class Market:
    def __init__(self):
        self.resources = {'wood': 0, 'stone': 0, 'food': 0}
        self.tools = {'pickaxe': [], 'axe': []}
        self.food_price = 1

    def add_tool(self, tool):
        self.tools[tool.tool_type].append(tool)

    def update_prices(self):
        self.food_price = max(1, 1 + (self.resources['food'] // 20))

# 城市类
class City:
    def __init__(self):
        self.resources = {
            'wood': Resource('wood', 100),
            'stone': Resource('stone', 100),
            'food': Resource('food', 1000)
        }
        self.citizens = []
        self.market = Market()
        self.time = 0

    def add_citizen(self, citizen):
        self.citizens.append(citizen)

    # 修改City类的step方法，大幅增加食物生产效率
    # 修改City类的step方法，进一步减少食物消耗和增加食物生产
    def step(self):
        self.time += 1
        self.market.update_prices()
        
        for citizen in self.citizens:
            if not citizen.alive:
                continue
                
            citizen.age += 1
            # 减少食物消耗速度，每两个时间步骤消耗一个食物
            if self.time % 3 == 0:
                citizen.food = max(0, citizen.food - 1)
            
            # 饥饿死亡机制
            if citizen.food <= 0:
                citizen.alive = False
                continue
                
            # 年龄死亡
            if citizen.age > 80:
                citizen.alive = False
        
        # 移除死亡居民
        self.citizens = [c for c in self.citizens if c.alive]
        
        # 大幅增加农民生产食物的效率
        food_produced = sum([5 for c in self.citizens if c.profession == 'farmer' and c.age >= 16])
        self.resources['food'].add(food_produced)
    
        # 工具维护
        for citizen in self.citizens:
            if citizen.tool and citizen.tool.durability <= 0:
                citizen.tool = None

# 环境类
class CityEnv:
    def __init__(self):
        self.city = City()
        self.action_size = 5  # 对应Citizen.action_space的长度
        self.state_size = 7   # 对应Citizen.get_state的返回维度
        
        # 初始化居民
        for i in range(30):
            citizen = Citizen(
                id=i,
                gender='male' if i%2 == 0 else 'female',
                age=random.randint(18, 60),
                wealth=150,
                food=150
            )
            self.city.add_citizen(citizen)

        # 增加初始资源
        self.city.resources['food'].quantity = 2000  # 从1000增加到2000
        self.city.resources['wood'].quantity = 200   # 从100增加到200
        self.city.resources['stone'].quantity = 200  # 从100增加到200    
        
        # 每个居民有自己的DQN
        self.models = [DQN(self.state_size, self.action_size) for _ in self.city.citizens]
        self.target_models = [DQN(self.state_size, self.action_size) for _ in self.city.citizens]
        for i in range(len(self.target_models)):
            self.target_models[i].load_state_dict(self.models[i].state_dict())

    # 修改CityEnv类的step方法，确保actions和alive_citizens长度一致
    # 修改CityEnv类的step方法中的奖励机制
    def step(self, actions):
        rewards = []
        alive_citizens = [c for c in self.city.citizens if c.alive]
        
        # 确保actions和alive_citizens长度一致
        if len(actions) != len(alive_citizens):
            # 如果actions少于alive_citizens，为剩余居民随机生成动作
            while len(actions) < len(alive_citizens):
                actions.append(random.randrange(self.action_size))
            # 如果actions多于alive_citizens，截断actions
            actions = actions[:len(alive_citizens)]
        
        # 计算当前各职业的比例
        profession_counts = {}
        for c in self.city.citizens:
            if c.alive and c.profession:
                profession_counts[c.profession] = profession_counts.get(c.profession, 0) + 1
        
        total_citizens = len(alive_citizens)
        farmer_ratio = profession_counts.get('farmer', 0) / total_citizens if total_citizens > 0 else 0
        
        # 全局奖励：基于城市状态的奖励，所有居民共享
        global_reward = 0
        
        # 人口奖励大幅增加
        if total_citizens > 20:
            global_reward += 3.0
        if total_citizens > 30:
            global_reward += 5.0
        if total_citizens > 40:
            global_reward += 8.0
        
        # 食物储备奖励
        if self.city.resources['food'].quantity > 500:
            global_reward += 2.0
        if self.city.resources['food'].quantity > 1000:
            global_reward += 3.0
        
        # 职业平衡奖励
        if 0.3 <= farmer_ratio <= 0.5:  # 理想的农民比例
            global_reward += 2.0
        
        for i, citizen in enumerate(alive_citizens):
            action = actions[i]
            reward = global_reward  # 从全局奖励开始
            
            # 年龄限制工作
            if citizen.age < 16 and citizen.action_space[action] == 'work':
                reward -= 1  # 惩罚未成年工作
                action = 4  # 改为idle
            
            # 执行动作
            if citizen.action_space[action] == 'work':
                if citizen.profession:  # 确保有职业
                    citizen.work(self.city)
                    reward += 5.0  # 大幅增加工作奖励
                    
                    # 根据职业需求给予额外奖励
                    if citizen.profession == 'farmer' and farmer_ratio < 0.4:
                        reward += 3.0
                else:
                    reward -= 0.5  # 惩罚无职业工作
            
            elif citizen.action_space[action] == 'buy_food':
                if citizen.wealth >= self.city.market.food_price and self.city.resources['food'].quantity > 0:
                    citizen.wealth -= self.city.market.food_price
                    self.city.resources['food'].subtract(1)
                    citizen.food += 1
                    
                    # 根据食物状况给予奖励
                    if citizen.food < 5:  # 如果食物不足，奖励更高
                        reward += 3.0
                    else:
                        reward += 1.0
                else:
                    reward -= 0.5  # 减轻惩罚
            
            elif citizen.action_space[action] == 'reproduce':
                if citizen.reproduce(self.city):
                    reward += 15.0  # 大幅增加繁衍奖励
                else:
                    reward -= 0.1  # 进一步减轻惩罚
            
            elif citizen.action_space[action] == 'change_profession':
                new_prof = random.choice(['lumberjack', 'miner', 'farmer', 'blacksmith', 'merchant'])
                cost = 5  # 职业变更需要花费
                
                if citizen.wealth >= cost:
                    citizen.wealth -= cost
                    old_prof = citizen.profession
                    citizen.profession = new_prof
                    
                    # 根据城市需求奖励合理职业选择
                    if new_prof == 'farmer' and farmer_ratio < 0.4:
                        reward += 5.0  # 大幅提高农民职业的奖励
                    elif new_prof == 'blacksmith' and len([c for c in self.city.citizens if c.tool is None]) > 5:
                        reward += 3.0
                    elif old_prof == 'farmer' and farmer_ratio > 0.5:  # 如果农民太多，允许转职
                        reward += 2.0
                    else:
                        reward += 1.0
                else:
                    reward -= 0.5  # 减轻惩罚
            
            # 为孩子购买食物，增加奖励
            children = [c for c in self.city.citizens if c.alive and c.age < 16]
            if children and citizen.age >= 16:
                for child in children:
                    if child.food < 3 and citizen.wealth >= self.city.market.food_price:
                        citizen.wealth -= self.city.market.food_price
                        child.food += 1
                        reward += 2.0  # 增加照顾孩子的奖励
            
            # 尝试购买工具
            if citizen.tool is None and (citizen.profession == 'lumberjack' or citizen.profession == 'miner'):
                if citizen.buy_tool(self.city):
                    reward += 3.0  # 增加购买工具的奖励
            
            # 资源生产奖励
            if citizen.action_space[action] == 'work':
                if 'wood' in self.city.resources:
                    reward += 0.1 * self.city.resources['wood'].quantity / 100
                if 'stone' in self.city.resources:
                    reward += 0.1 * self.city.resources['stone'].quantity / 100
                if 'food' in self.city.resources:
                    reward += 0.3 * self.city.resources['food'].quantity / 100  # 增加食物奖励权重
            
            # 生存奖励
            if citizen.age < 80 and citizen.food > 3:
                reward += 3.0  # 大幅增加生存奖励
                
            # 工具使用效率奖励
            if citizen.tool and citizen.tool.durability > 0:
                reward += 1.0  # 增加工具使用奖励
                
            # 惩罚无效行为
            if citizen.action_space[action] == 'idle':
                reward -= 0.2  # 进一步减轻惩罚

            rewards.append(reward)
            
        self.city.step()
        next_states = self.get_states()
        done = self.is_done()
        
        # 确保状态数量一致
        alive_citizens_new = [c for c in self.city.citizens if c.alive]
        if len(alive_citizens) != len(alive_citizens_new):
            next_states = [c.get_state(self.city) for c in alive_citizens]
                
        return next_states, rewards, done

    def get_states(self):
        return [citizen.get_state(self.city) for citizen in self.city.citizens if citizen.alive]

    def is_done(self):
        return len(self.city.citizens) == 0 or self.city.resources['food'].quantity <= 0

# DQN模型
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.fc(x)

# 训练参数
env = CityEnv()
memory = deque(maxlen=10000)
gamma = 0.98
epsilon = 1.0
epsilon_min = 0.01
# 修改训练参数，降低探索率的衰减速度，让智能体有更多机会探索
epsilon_decay = 0.9995  # 从0.995改为0.998，减缓探索率下降
batch_size = 32
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in env.models]

# 修改act函数
# 修改act函数，确保为每个活着的居民生成动作
def act(states):
    actions = []
    alive_citizens = [c for c in env.city.citizens if c.alive]
    
    # 确保states和alive_citizens长度一致
    if len(states) != len(alive_citizens):
        # 如果不一致，只处理两者中较小的数量
        min_len = min(len(states), len(alive_citizens))
        states = states[:min_len]
        alive_citizens = alive_citizens[:min_len]
    
    for i, state in enumerate(states):
        if random.random() < epsilon:
            actions.append(random.randrange(env.action_size))
        else:
            with torch.no_grad():
                q_values = env.models[i](torch.FloatTensor(state))
                actions.append(q_values.argmax().item())
    
    return actions

def replay(batch_size):
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    
    for transition in minibatch:
        states, actions, rewards, next_states, done = transition
        
        for i in range(len(states)):
            # 确保索引不超出模型数量范围
            if i >= len(env.models) or i >= len(env.target_models):
                continue
                
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i] if i < len(next_states) else state  # 防止越界
            
            target = reward
            if not done:
                with torch.no_grad():
                    target = reward + gamma * env.target_models[i](
                        torch.FloatTensor(next_state)
                    ).max().item()
            
            target_f = env.models[i](torch.FloatTensor(state)).detach().numpy()
            target_f[action] = target
            
            env.models[i].zero_grad()
            loss = nn.MSELoss()(env.models[i](torch.FloatTensor(state)),
                               torch.FloatTensor(target_f))
            loss.backward()
            optimizers[i].step()

# 修改训练循环，确保模型数量与居民数量同步
episodes = 1000
for e in range(episodes):
    # 确保模型数量与居民数量匹配
    alive_count = len([c for c in env.city.citizens if c.alive])
    
    # 调整模型数量
    while len(env.models) < alive_count:
        new_model = DQN(env.state_size, env.action_size)
        env.models.append(new_model)
        env.target_models.append(DQN(env.state_size, env.action_size))
        env.target_models[-1].load_state_dict(new_model.state_dict())
        optimizers.append(optim.Adam(new_model.parameters(), lr=0.001))
    
    # 如果模型过多，截断多余的模型和优化器
    if len(env.models) > alive_count:
        env.models = env.models[:alive_count]
        env.target_models = env.target_models[:alive_count]
        optimizers = optimizers[:alive_count]
    
    # 确保优化器数量与模型数量一致
    while len(optimizers) < len(env.models):
        optimizers.append(optim.Adam(env.models[len(optimizers)].parameters(), lr=0.001))
    if len(optimizers) > len(env.models):
        optimizers = optimizers[:len(env.models)]
    
    states = env.get_states()
    total_rewards = 0
    done = False
    
    while not done:
        actions = act(states)
        next_states, rewards, done = env.step(actions)
        
        # 处理居民数量变化
        if len(states) != len(next_states):
            alive_count = min(len(states), len(next_states))
            states_padded = states[:alive_count]
            actions_padded = actions[:alive_count]
            rewards_padded = rewards[:alive_count]
        else:
            states_padded = states
            actions_padded = actions
            rewards_padded = rewards
            
        memory.append((states_padded, actions_padded, rewards_padded, next_states, done))
        
        states = next_states
        total_rewards += sum(rewards)
        
        replay(batch_size)

        # 每10步打印一次状态，减少输出量
        if env.city.time % 10 == 0:
            print(f"Time: {env.city.time}, Alive: {len(env.city.citizens)}, " 
                  f"Food: {env.city.resources['food'].quantity}, "
                  f"Wealth: {sum([c.wealth for c in env.city.citizens])}")
        
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        
    if e % 10 == 0:
        # 只更新存在的模型
        for i in range(min(len(env.models), len(env.target_models))):
            env.target_models[i].load_state_dict(env.models[i].state_dict())
            
    print(f"Episode: {e}/{episodes}, Total Reward: {total_rewards:.2f}, Epsilon: {epsilon:.2f}")