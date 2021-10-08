import numpy as np
import gym

class Env_2048(gym.Env):
    metadata = {'render.modes':['human']}
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=2**17, shape=(4,4))
        self.action_space = gym.spaces.Discrete(4)
        self.actions = [self.up,self.right,self.down,self.left]
        #self._seed(seed)
        self.reset()

    def reset(self):
        self.table = np.zeros((4,4), dtype=int)
        self.initial_place()
        return self.table

    def step(self, action):
        assert action == 0 or action == 1 or action == 2 or action == 3, 'action must be 0~3'
        done = False
        afterstate, reward = self.actions[action]()
        if not np.array_equal(self.table, afterstate):
            self.table = afterstate.copy()
            self.add_number()
        if self.is_done():
            done = True
        return self.table.copy(), reward, done, afterstate
        #return self.table.copy(), reward, done, None
    
    def render(self, mode='human'):
        print(self.table)
    
    def observe(self):
        return self.table
    
    def initial_place(self):
        self.add_number()
        self.add_number()

    def two_or_four(self):
        return 2 if np.random.random() < 0.9 else 4
    
    def add_number(self):
        row, column = np.where(self.table==0)
        if row.size == 0:
            raise Exception("Error")
        n = np.random.choice(range(row.size))
        self.table[row[n]][column[n]] = self.two_or_four()

    def is_done(self):
        if np.any(self.table==0):
            return False
        if np.any(np.diff(self.table, axis=0) == 0):
            return False
        if np.any(np.diff(self.table, axis=1) == 0):
            return False
        return True
    
    def right(self):
        table_tmp = self.table.copy()
        double_list = []
        reward = 0
        for i in range(4):
            for j in range(2, -1, -1):
                if table_tmp[i][j] == 0:
                    continue
                for k in range(j+1, 4):
                    if table_tmp[i][k] == 0 and k == 3:
                        table_tmp[i][k] = table_tmp[i][j]
                        table_tmp[i][j] = 0
                    elif table_tmp[i][k] == 0 and k != 3:
                        continue
                    elif table_tmp[i][k] == table_tmp[i][j]:
                        double_list.append((i,k,table_tmp[i][k]*2))
                        table_tmp[i][k] = -1
                        table_tmp[i][j] = 0
                        break
                    else:
                        if j == k-1:
                            break
                        else:
                            table_tmp[i][k-1] = table_tmp[i][j]
                            table_tmp[i][j] = 0
                            break
        for double in double_list:
            table_tmp[double[0]][double[1]] = double[2]
            reward += double[2]
        return table_tmp, reward
    
    def left(self):
        table_tmp = self.table.copy()
        double_list = []
        reward = 0
        for i in range(4):
            for j in range(1, 4):
                if table_tmp[i][j] == 0:
                    continue
                for k in range(j-1, -1, -1):
                    if table_tmp[i][k] == 0 and k == 0:
                        table_tmp[i][k] = table_tmp[i][j]
                        table_tmp[i][j] = 0
                    elif table_tmp[i][k] == 0 and k != 0:
                        continue
                    elif table_tmp[i][k] == table_tmp[i][j]:
                        double_list.append((i,k,table_tmp[i][k]*2))
                        table_tmp[i][k] = -1
                        table_tmp[i][j] = 0
                        break
                    else:
                        if j == k+1:
                            break
                        else:
                            table_tmp[i][k+1] = table_tmp[i][j]
                            table_tmp[i][j] = 0
                            break
        for double in double_list:
            table_tmp[double[0]][double[1]] = double[2]
            reward += double[2]
        return table_tmp, reward

    def up(self):
        table_tmp = self.table.copy()
        double_list = []
        reward = 0
        for j in range(4):
            for i in range(1, 4):
                if table_tmp[i][j] == 0:
                    continue
                for k in range(i-1, -1, -1):
                    if table_tmp[k][j] == 0 and k == 0:
                        table_tmp[k][j] = table_tmp[i][j]
                        table_tmp[i][j] =0 
                    elif table_tmp[k][j] == 0 and k != 0:
                        continue
                    elif table_tmp[k][j] == table_tmp[i][j]:
                        double_list.append((k,j,table_tmp[k][j]*2))
                        table_tmp[k][j] = -1
                        table_tmp[i][j] = 0
                        break
                    else:
                        if i == k+1:
                            break
                        else:
                            table_tmp[k+1][j] = table_tmp[i][j]
                            table_tmp[i][j] = 0
                            break
        for double in double_list:
            table_tmp[double[0]][double[1]] = double[2]
            reward += double[2]
        return table_tmp, reward
    
    def down(self):
        table_tmp = self.table.copy()
        double_list = []
        reward = 0
        for j in range(4):
            for i in range(2, -1, -1):
                if table_tmp[i][j] == 0:
                    continue
                for k in range(i+1, 4):
                    if table_tmp[k][j] == 0 and k == 3:
                        table_tmp[k][j] = table_tmp[i][j]
                        table_tmp[i][j] = 0
                    elif table_tmp[k][j] == 0 and k != 3:
                        continue
                    elif table_tmp[k][j] == table_tmp[i][j]:
                        double_list.append((k,j,table_tmp[k][j]*2))
                        table_tmp[k][j] = -1
                        table_tmp[i][j] = 0
                        break
                    else:
                        if i == k-1:
                            break
                        else:
                            table_tmp[k-1][j] = table_tmp[i][j]
                            table_tmp[i][j] = 0
                            break
        for double in double_list:
            table_tmp[double[0]][double[1]] = double[2]
            reward += double[2]
        return table_tmp, reward
    
    def _seed(self, seed):
        np.random.seed(seed)

if __name__ == '__main__':
    env = Env_2048()
    print('Game Start!')
    env.reset()
    done = False
    while not done:
        env.render()
        action = int(input('select from above: '))
        env.step(action)