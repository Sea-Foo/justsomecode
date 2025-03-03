import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class Resource:
    def __init__(self, name, quantity, position):
        self.name = name
        self.quantity = quantity
        self.position = position  # (x, y) 坐标
        self.regeneration_rate = 0.1  # 资源再生率

    def add(self, amount):
        self.quantity = max(0, self.quantity + amount)

    def subtract(self, amount):
        if self.quantity >= amount:
            self.quantity -= amount
            return True
        return False

    def regenerate(self):
        if self.name in ['wood', 'stone', 'gold']:
            self.quantity += self.regeneration_rate

class Tool:
    def __init__(self, tool_type, durability):
        self.tool_type = tool_type
        self.durability = durability

    def use(self):
        self.durability -= 1
        return self.durability > 0

class Market:
    def __init__(self):
        self.resources = {'wood': 0, 'stone': 0, 'gold': 0, 'food': 0}
        self.tools = {'pickaxe': [], 'axe': []}
        self.prices = {
            'wood': 1,
            'stone': 2,
            'gold': 5,
            'food': 1,
            'pickaxe': 10,
            'axe': 10
        }

    def add_tool(self, tool):
        self.tools[tool.tool_type].append(tool)

    def update_prices(self):
        # 根据供需关系动态调整价格
        for resource, quantity in self.resources.items():
            base_price = self.prices[resource]
            if quantity > 100:
                self.prices[resource] = max(1, base_price * 0.9)
            elif quantity < 20:
                self.prices[resource] = base_price * 1.1

class Citizen:
    def __init__(self, id, gender, age, wealth, food, position):
        self.id = id
        self.gender = gender
        self.age = age
        self.wealth = wealth
        self.food = food
        self.position = position  # (x, y) 坐标
        self.tool = None
        self.alive = True
        self.profession = random.choice(['farmer', 'farmer', 'farmer', 'lumberjack', 'miner', 'goldminer', 'blacksmith', 'merchant'])
        self.work_efficiency = random.uniform(0.8, 1.2)  # 工作效率属性
        self.action_space = ['work', 'buy_food', 'reproduce', 'change_profession', 'move', 'idle']

    def get_state(self, city):
        nearby_resources = city.get_nearby_resources(self.position)
        return np.array([
            self.age,
            self.wealth,
            self.food,
            city.market.resources['food'],  # 使用市场中的食物数量
            city.market.prices['food'],
            len([c for c in city.citizens if c.alive]),
            self.tool.durability if self.tool else 0,
            *nearby_resources,  # 附近资源的状态
            self.position[0] / city.map_size[0],  # 归一化的位置
            self.position[1] / city.map_size[1]
        ], dtype=np.float32)

    def select_action(self, model, epsilon, city):
        if random.random() < epsilon:
            return random.choice(range(len(self.action_space)))
        
        with torch.no_grad():
            state = torch.FloatTensor(self.get_state(city)).unsqueeze(0)
            q_values = model(state)
            return q_values.argmax().item()

    def move(self, city):
        # 在地图上移动
        dx = random.randint(-1, 1)
        dy = random.randint(-1, 1)
        new_x = max(0, min(city.map_size[0] - 1, self.position[0] + dx))
        new_y = max(0, min(city.map_size[1] - 1, self.position[1] + dy))
        self.position = (new_x, new_y)

    def work(self, city):
        if self.profession == 'lumberjack':
            nearby_wood = city.find_nearest_resource(self.position, 'wood')
            if nearby_wood:
                efficiency = self.work_efficiency
                if self.tool and self.tool.tool_type == 'axe':
                    efficiency *= 2
                    self.tool.use()
                amount = efficiency
                if nearby_wood.subtract(amount):
                    city.market.resources['wood'] += amount
                    self.wealth += amount * city.market.prices['wood']

        elif self.profession == 'miner':
            nearby_stone = city.find_nearest_resource(self.position, 'stone')
            if nearby_stone:
                efficiency = self.work_efficiency
                if self.tool and self.tool.tool_type == 'pickaxe':
                    efficiency *= 2
                    self.tool.use()
                amount = efficiency
                if nearby_stone.subtract(amount):
                    city.market.resources['stone'] += amount
                    self.wealth += amount * city.market.prices['stone']

        elif self.profession == 'goldminer':
            nearby_gold = city.find_nearest_resource(self.position, 'gold')
            if nearby_gold:
                efficiency = self.work_efficiency
                if self.tool and self.tool.tool_type == 'pickaxe':
                    efficiency *= 2
                    self.tool.use()
                amount = efficiency * 0.5  # 金矿开采较慢
                if nearby_gold.subtract(amount):
                    city.market.resources['gold'] += amount
                    self.wealth += amount * city.market.prices['gold']

        elif self.profession == 'farmer':
            efficiency = self.work_efficiency * 5
            city.market.resources['food'] += efficiency
            self.wealth += efficiency * city.market.prices['food']

        elif self.profession == 'blacksmith':
            if city.market.resources['wood'] >= 1 and city.market.resources['stone'] >= 2:
                city.market.resources['wood'] -= 1
                city.market.resources['stone'] -= 2
                tool_type = random.choice(['axe', 'pickaxe'])
                new_tool = Tool(tool_type, 10)
                city.market.add_tool(new_tool)
                self.wealth += 5

        elif self.profession == 'merchant':
            # 商人通过买卖资源赚取差价
            for resource in ['wood', 'stone', 'gold']:
                if city.market.resources[resource] > 50:
                    profit = city.market.prices[resource] * 0.2
                    self.wealth += profit

class City:
    def __init__(self, map_size=(20, 20)):
        self.map_size = map_size
        self.resources = {}
        self.citizens = []
        self.market = Market()
        self.time = 0
        self.initialize_resources()

    def initialize_resources(self):
        # 在地图上随机分布资源
        for resource_type in ['wood', 'stone', 'gold']:
            for _ in range(5):  # 每种资源创建多个点
                pos = (random.randint(0, self.map_size[0]-1),
                      random.randint(0, self.map_size[1]-1))
                quantity = random.randint(50, 100)
                self.resources[pos] = Resource(resource_type, quantity, pos)

    def get_nearby_resources(self, position, radius=2):
        # 获取指定位置附近的资源状态
        nearby = [0, 0, 0]  # wood, stone, gold
        for resource in self.resources.values():
            dx = abs(position[0] - resource.position[0])
            dy = abs(position[1] - resource.position[1])
            if dx <= radius and dy <= radius:
                if resource.name == 'wood':
                    nearby[0] += resource.quantity
                elif resource.name == 'stone':
                    nearby[1] += resource.quantity
                elif resource.name == 'gold':
                    nearby[2] += resource.quantity
        return nearby

    def find_nearest_resource(self, position, resource_type):
        nearest = None
        min_dist = float('inf')
        for resource in self.resources.values():
            if resource.name == resource_type:
                dist = abs(position[0] - resource.position[0]) + \
                       abs(position[1] - resource.position[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest = resource
        return nearest

    def step(self):
        self.time += 1
        self.market.update_prices()

        # 资源再生
        for resource in self.resources.values():
            resource.regenerate()

        # 居民行动
        for citizen in self.citizens:
            if not citizen.alive:
                continue

            citizen.age += 1
            if self.time % 3 == 0:
                citizen.food = max(0, citizen.food - 1)

            # 饥饿或年龄导致的死亡
            if citizen.food <= 0 or citizen.age > 80:
                citizen.alive = False
                continue

        # 移除死亡居民
        self.citizens = [c for c in self.citizens if c.alive]

class CityEnv:
    def __init__(self):
        self.city = City()
        self.action_size = 6  # work, buy_food, reproduce, change_profession, move, idle
        self.state_size = 12  # 包括位置和周围资源信息

        # 初始化居民
        for i in range(30):
            position = (random.randint(0, self.city.map_size[0]-1),
                       random.randint(0, self.city.map_size[1]-1))
            citizen = Citizen(
                id=i,
                gender='male' if i%2 == 0 else 'female',
                age=random.randint(18, 60),
                wealth=150,
                food=150,
                position=position
            )
            self.city.citizens.append(citizen)

        # 每个居民的DQN
        self.models = [DQN(self.state_size, self.action_size) for _ in self.city.citizens]
        self.target_models = [DQN(self.state_size, self.action_size) for _ in self.city.citizens]
        for i in range(len(self.target_models)):
            self.target_models[i].load_state_dict(self.models[i].state_dict())

    def step(self, actions):
        rewards = []
        alive_citizens = [c for c in self.city.citizens if c.alive]

        if len(actions) != len(alive_citizens):
            while len(actions) < len(alive_citizens):
                actions.append(random.randrange(self.action_size))
            actions = actions[:len(alive_citizens)]

        # 计算当前各职业的比例
        profession_counts = {}
        for c in alive_citizens:
            profession_counts[c.profession] = profession_counts.get(c.profession, 0) + 1

        total_citizens = len(alive_citizens)
        farmer_ratio = profession_counts.get('farmer', 0) / total_citizens if total_citizens > 0 else 0

        # 全局奖励
        global_reward = 0
        if total_citizens > 20:
            global_reward += 3.0
        if total_citizens > 30:
            global_reward += 5.0
        if total_citizens > 40:
            global_reward += 8.0

        if self.city.market.resources['food'] > 500:
            global_reward += 2.0

        if 0.3 <= farmer_ratio <= 0.5:
            global_reward += 2.0

        for i, citizen in enumerate(alive_citizens):
            action = actions[i]
            reward = global_reward

            # 执行动作
            if citizen.action_space[action] == 'work':
                if citizen.age >= 16:
                    citizen.work(self.city)
                    reward += 5.0
                    if citizen.profession == 'farmer' and farmer_ratio < 0.4:
                        reward += 3.0
                else:
                    reward -= 1.0

            elif citizen.action_space[action] == 'move':
                old_pos = citizen.position
                citizen.move(self.city)
                if self.city.find_nearest_resource(citizen.position, 
                    'wood' if citizen.profession == 'lumberjack' else
                    'stone' if citizen.profession == 'miner' else
                    'gold' if citizen.profession == 'goldminer' else None):
                    reward += 2.0

            elif citizen.action_space[action] == 'buy_food':
                if citizen.wealth >= self.city.market.prices['food']:
                    citizen.wealth -= self.city.market.prices['food']
                    citizen.food += 1
                    if citizen.food < 5:
                        reward += 3.0
                    else:
                        reward += 1.0
                else:
                    reward -= 0.5

            elif citizen.action_space[action] == 'reproduce':
                if citizen.reproduce(self.city):
                    reward += 10.0  # 繁衍成功给予高奖励
                else:
                    reward -= 0.1
            
            elif citizen.action_space[action] == 'change_profession':
                old_prof = citizen.profession
                new_prof = random.choice(['farmer', 'lumberjack', 'miner', 'goldminer', 'blacksmith', 'merchant'])
                citizen.profession = new_prof
                
                # 根据城市需求奖励合理职业选择
                if new_prof == 'farmer' and farmer_ratio < 0.4:
                    reward += 5.0
                elif new_prof == 'blacksmith' and len([c for c in self.city.citizens if c.tool is None]) > 5:
                    reward += 3.0
                elif old_prof == 'farmer' and farmer_ratio > 0.5:
                    reward += 2.0
                else:
                    reward += 1.0
            
            # 为孩子购买食物
            children = [c for c in self.city.citizens if c.alive and c.age < 16]
            if children and citizen.age >= 16:
                for child in children:
                    if child.food < 3 and citizen.wealth >= self.city.market.prices['food']:
                        citizen.wealth -= self.city.market.prices['food']
                        child.food += 1
                        reward += 2.0
            
            rewards.append(reward)

        self.city.step()
        next_states = self.get_states()
        done = self.is_done()

        return next_states, rewards, done

    def get_states(self):
        return [citizen.get_state(self.city) for citizen in self.city.citizens if citizen.alive]

    def is_done(self):
        return len(self.city.citizens) == 0

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

def train_citizens(env, num_episodes=1000, batch_size=64, gamma=0.99):
    # 为每个居民创建经验回hfill区
    memory_size = 10000
    memories = [ReplayBuffer(memory_size) for _ in range(len(env.city.citizens))]
    
    # 优化器
    optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in env.models]
    
    # 探索率
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.995
    epsilon = epsilon_start
    
    for episode in range(num_episodes):
        # 获取当前状态
        states = env.get_states()
        total_rewards = [0] * len(states)
        
        # 每个episode最多运行1000步
        for step in range(1000):
            # 选择动作
            actions = []
            for i, citizen in enumerate([c for c in env.city.citizens if c.alive]):
                action = citizen.select_action(env.models[i], epsilon, env.city)
                actions.append(action)
            
            # 执行动作
            next_states, rewards, done = env.step(actions)
            
            # 存储经验
            for i in range(min(len(states), len(next_states))):
                memories[i].push(states[i], actions[i], rewards[i], next_states[i], done)
                total_rewards[i] += rewards[i]
            
            # 如果游戏结束，跳出循环
            if done:
                break
                
            # 从经验回放中学习
            for i in range(len(env.models)):
                if len(memories[i]) > batch_size:
                    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = memories[i].sample(batch_size)
                    
                    # 转换为tensor
                    batch_states = torch.FloatTensor(batch_states)
                    batch_actions = torch.LongTensor(batch_actions)
                    batch_rewards = torch.FloatTensor(batch_rewards)
                    batch_next_states = torch.FloatTensor(batch_next_states)
                    batch_dones = torch.FloatTensor(batch_dones)
                    
                    # 计算当前Q值
                    current_q = env.models[i](batch_states).gather(1, batch_actions.unsqueeze(1))
                    
                    # 计算目标Q值
                    with torch.no_grad():
                        max_next_q = env.target_models[i](batch_next_states).max(1)[0]
                        target_q = batch_rewards + (1 - batch_dones) * gamma * max_next_q
                    
                    # 计算损失
                    loss = nn.MSELoss()(current_q.squeeze(), target_q)
                    
                    # 优化模型
                    optimizers[i].zero_grad()
                    loss.backward()
                    optimizers[i].step()
            
            # 更新状态
            states = next_states
        
        # 更新目标网络
        if episode % 10 == 0:
            for i in range(len(env.models)):
                env.target_models[i].load_state_dict(env.models[i].state_dict())
        
        # 衰减探索率
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 打印进度
        if episode % 10 == 0:
            avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
            print(f"Episode {episode}, Average Reward: {avg_reward}, Population: {len(env.city.citizens)}")

# 添加繁衍方法到Citizen类
def reproduce(self, city):
    # 只有女性、适龄且有足够食物的居民可以繁衍
    if self.gender == 'male' or self.age < 18 or self.age > 50 or self.food < 2:
        return False
    
    # 寻找合适的伴侣
    partners = [c for c in city.citizens if c.alive and c.gender == 'male' 
                and 18 <= c.age <= 60 and c.food >= 2]
    
    if not partners:
        return False
    
    partner = random.choice(partners)
    
    # 消耗资源
    self.food -= 1
    partner.food -= 1
    
    # 创建新居民
    child_id = max([c.id for c in city.citizens]) + 1 if city.citizens else 0
    child_gender = random.choice(['male', 'female'])
    position = (random.randint(max(0, self.position[0]-1), min(city.map_size[0]-1, self.position[0]+1)),
               random.randint(max(0, self.position[1]-1), min(city.map_size[1]-1, self.position[1]+1)))
    
    child = Citizen(
        id=child_id,
        gender=child_gender,
        age=0,  # 新生儿
        wealth=10,  # 给予一些初始财富
        food=10,  # 初始食物
        position=position
    )
    
    city.citizens.append(child)
    return True

# 将繁衍方法添加到Citizen类
Citizen.reproduce = reproduce

# 修改CityEnv的step方法，添加繁衍和职业变更的处理
def enhanced_step(self, actions):
    rewards = []
    alive_citizens = [c for c in self.city.citizens if c.alive]
    
    if len(actions) != len(alive_citizens):
        while len(actions) < len(alive_citizens):
            actions.append(random.randrange(self.action_size))
        actions = actions[:len(alive_citizens)]
    
    # 计算当前各职业的比例
    profession_counts = {}
    for c in alive_citizens:
        profession_counts[c.profession] = profession_counts.get(c.profession, 0) + 1
    
    total_citizens = len(alive_citizens)
    farmer_ratio = profession_counts.get('farmer', 0) / total_citizens if total_citizens > 0 else 0
    
    # 全局奖励
    global_reward = 0
    if total_citizens > 20:
        global_reward += 3.0
    if total_citizens > 30:
        global_reward += 5.0
    if total_citizens > 40:
        global_reward += 8.0
    
    if self.city.market.resources['food'] > 500:
        global_reward += 2.0
    
    if 0.3 <= farmer_ratio <= 0.5:
        global_reward += 2.0
    
    for i, citizen in enumerate(alive_citizens):
        action = actions[i]
        reward = global_reward
        
        # 执行动作
        if citizen.action_space[action] == 'work':
            if citizen.age >= 16:
                citizen.work(self.city)
                reward += 5.0
                if citizen.profession == 'farmer' and farmer_ratio < 0.4:
                    reward += 3.0
            else:
                reward -= 1.0
        
        elif citizen.action_space[action] == 'move':
            old_pos = citizen.position
            citizen.move(self.city)
            if self.city.find_nearest_resource(citizen.position, 
                'wood' if citizen.profession == 'lumberjack' else
                'stone' if citizen.profession == 'miner' else
                'gold' if citizen.profession == 'goldminer' else None):
                reward += 2.0
        
        elif citizen.action_space[action] == 'buy_food':
            if citizen.wealth >= self.city.market.prices['food']:
                citizen.wealth -= self.city.market.prices['food']
                citizen.food += 1
                if citizen.food < 5:
                    reward += 3.0
                else:
                    reward += 1.0
            else:
                reward -= 0.5
        
        elif citizen.action_space[action] == 'reproduce':
            if citizen.reproduce(self.city):
                reward += 10.0  # 繁衍成功给予高奖励
            else:
                reward -= 0.1
        
        elif citizen.action_space[action] == 'change_profession':
            old_prof = citizen.profession
            new_prof = random.choice(['farmer', 'lumberjack', 'miner', 'goldminer', 'blacksmith', 'merchant'])
            citizen.profession = new_prof
            
            # 根据城市需求奖励合理职业选择
            if new_prof == 'farmer' and farmer_ratio < 0.4:
                reward += 5.0
            elif new_prof == 'blacksmith' and len([c for c in self.city.citizens if c.tool is None]) > 5:
                reward += 3.0
            elif old_prof == 'farmer' and farmer_ratio > 0.5:
                reward += 2.0
            else:
                reward += 1.0
        
        # 为孩子购买食物
        children = [c for c in self.city.citizens if c.alive and c.age < 16]
        if children and citizen.age >= 16:
            for child in children:
                if child.food < 3 and citizen.wealth >= self.city.market.prices['food']:
                    citizen.wealth -= self.city.market.prices['food']
                    child.food += 1
                    reward += 2.0
        
        rewards.append(reward)
    
    self.city.step()
    next_states = self.get_states()
    done = self.is_done()
    
    return next_states, rewards, done

# 替换CityEnv的step方法
CityEnv.step = enhanced_step

# 主函数
if __name__ == "__main__":
    env = CityEnv()
    train_citizens(env, num_episodes=500)
    
    # 训练后测试
    print("\nTesting trained citizens...")
    states = env.get_states()
    total_population = []
    
    for episode in range(100):
        for step in range(100):
            actions = []
            for i, citizen in enumerate([c for c in env.city.citizens if c.alive]):
                with torch.no_grad():
                    state = torch.FloatTensor(citizen.get_state(env.city)).unsqueeze(0)
                    q_values = env.models[i](state)
                    action = q_values.argmax().item()
                actions.append(action)
            
            next_states, rewards, done = env.step(actions)
            if done:
                break
            
            states = next_states
        
        total_population.append(len(env.city.citizens))
        print(f"Episode {episode}, Population: {len(env.city.citizens)}")
    
    print(f"Average population over 100 episodes: {sum(total_population)/len(total_population)}")