import ray
import time
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from random import randint, choice, uniform
import random
#%matplotlib inline
import pickle
import numpy as np
import tqdm

import sys
from contextlib import closing

from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize, precision = 2)
#TransitionProb = [0.7, 0.1, 0.1, 0.1]
#TransitionProb = [1, 0, 0, 0]
TransitionProb = [0.97, 0.01, 0.01, 0.01]
def generate_row(length, h_prob):
    row = np.random.choice(2, length, p=[1.0 - h_prob, h_prob])
    row = ''.join(list(map(lambda z: 'F' if z == 0 else 'H', row)))
    return row


def generate_map(shape):
    """

    :param shape: Width x Height
    :return: List of text based map
    """
    h_prob = 0.1
    grid_map = []

    for h in range(shape[1]):

        if h == 0:
            row = 'SF'
            row += generate_row(shape[0] - 2, h_prob)
        elif h == 1:
            row = 'FF'
            row += generate_row(shape[0] - 2, h_prob)

        elif h == shape[1] - 1:
            row = generate_row(shape[0] - 2, h_prob)
            row += 'FG'
        elif h == shape[1] - 2:
            row = generate_row(shape[0] - 2, h_prob)
            row += 'FF'
        else:
            row = generate_row(shape[0], h_prob)

        grid_map.append(row)
        del row

    return grid_map

MAPS = {

    "4x4": [
        "SFFF",
        "FHFH",
        "FFFF",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "Dangerous Hallway": [
        "SFFFFFFF",
        "FFFFFFFF",
        "HFHHHFFF",
        "HFHHHFFF",
        "HFHHHFFF",
        "HFHHHFFF",
        "HFFFFFFF",
        "FGFFFFFF"
    ],
    "16x16": [
        "SFFFFFFFFHFFFFHF",
        "FFFFFFFFFFFFFHFF",
        "FFFHFFFFHFFFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFFFFFFFFFFF",
        "FFHHFFFFFFFHFFFH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFHFFFFFFHFFF",
        "FFFFFHFFFFFFFFFH",
        "FFFFFFFHFFFFFFFF",
        "FFFFFFFFFFFFHFFF",
        "FFFFFFHFFFFFFFFF",
        "FFFFFFFFHFFFFFFF",
        "FFFFFFFFFHFFFFHF",
        "FFFFFFFFFFHFFFFF",
        "FFFHFFFFFFFFFFFG",
    ],

    "32x32": [
        'SFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFFFFFFHFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFHFFFFFFFFHFFHFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFHFHHFHFHFFFFFHFFFH',
        'FFFFHFFFFFFFFFFFFFFFHFHFFFFFFFHF',
        'FFFFFHFFFFFFFFFFHFFFFFFFFFFHFFFF',
        'FFHHFFFFHFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFHFFFFFFFFFFHFFFHFHFFFFFFFFHFF',
        'FFFFHFFFFFFHFFFFHFHFFFFFFFFFFFFH',
        'FFFFHHFHFFFFHFFFFFFFFFFFFFFFFFFF',
        'FHFFFFFFFFFFHFFFFFFFFFFFHHFFFHFH',
        'FFFHFFFHFFFFFFFFFFFFFFFFFFFFHFFF',
        'FFFHFHFFFFFFFFHFFFFFFFFFFFFHFFHF',
        'FFFFFFFFFFFFFFFFHFFFFFFFHFFFFFFF',
        'FFFFFFHFFFFFFFFHHFFFFFFFHFFFFFFF',
        'FFHFFFFFFFFFHFFFFFFFFFFHFFFFFFFF',
        'FFFHFFFFFFFFFHFFFFHFFFFFFHFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFFFHFFFFF',
        'FFFFFFFFHFFFFFFFHFFFFFFFFFFFFFFH',
        'FFHFFFFFFFFFFFFFFFHFFFFFFFFFFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFHFFFFFFFHFFF',
        'FFHFFFFHFFFFFFFFFHFFFFFFFFFFFHFH',
        'FFFFFFFFFFHFFFFHFFFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFHHFFHHHFFFHFFFF',
        'FFFFFFFFFFFFFFHFFFFHFFFFFFFHFFFF',
        'FFFFFFFHFFFFFFFFFFFFFFFFFFFFFFFF',
        'FFFFFHFFFFFFFFFFFFFFFFHFFHFFFFFF',
        'FFFFFFFHFFFFFFFFFHFFFFFFFFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFF',
        'FFFFFFFFFFFFFFFFFFFFFFFFHFFFFFFF',
        'FFFFFFFFFFFFFFFHFFFFFFFFHFFFFFFG',
    ]
}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # BFS to check that it's a valid path.
    def is_valid(arr, r=0, c=0):
        if arr[r][c] == 'G':
            return True

        tmp = arr[r][c]
        arr[r][c] = "#"

        # Recursively check in all four directions.
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for x, y in directions:
            r_new = r + x
            c_new = c + y
            if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                continue

            if arr[r_new][c_new] not in '#H':
                if is_valid(arr, r_new, c_new):
                    arr[r][c] = tmp
                    return True

        arr[r][c] = tmp
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]


class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following

        SFFF
        FHFH
        FFFH
        HFFG

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located

    The episode ends when you reach the goal or fall in a hole.

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4"):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol
        self.nS = nS

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        rew_hole = -1000
        rew_goal = 1000
        rew_step = -1

        exit = nrow * ncol
        P = {s : {a : [] for a in range(nA)} for s in range(nS + 1)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'H':
                        li.append((1.0, exit, -1000, True))
                    elif letter in b'G':
                        li.append((1.0, exit, 1000, True))
                    else:
                        for b, p in zip([a, (a+1)%4, (a+2)%4, (a+3)%4], TransitionProb):
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            rew = rew_step
                            li.append((p, newstate, rew, False))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        if self.s < self.nS:
            desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        else:
            outfile.write("exit\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()

#map_4 = (MAPS["4x4"], 4)
#map_8 = (MAPS["8x8"], 8)
#map_32 = (MAPS["32x32"], 32)
#map_50 = (generate_map((50,50)), 50)
#map_110 = (generate_map((110,110)), 110)

map_DH = (MAPS["Dangerous Hallway"], 8)
map_16 = (MAPS["16x16"], 16)

MAP = map_DH
map_size = MAP[1]
run_time = {}

def plot_result(total_rewards, learning_num, legend, figname):
    print("\nLearning Performance:\n")
    episodes = []
    for i in range(len(total_rewards)):
        episodes.append(i * learning_num + 1)

    plt.figure(num = 1)
    fig, ax = plt.subplots()
    plt.plot(episodes, total_rewards)
    plt.title('performance')
    plt.legend(legend)
    plt.xlabel("Episodes")
    plt.ylabel("total rewards")
    plt.savefig(figname)

def plot_image(q_table, MAP, figname):

    best_value = np.max(q_table, axis = 1)[:-1].reshape((map_size,map_size))
    best_policy = np.argmax(q_table, axis = 1)[:-1].reshape((map_size,map_size))

    print("\n\nBest Q-value and Policy:\n")
    fig, ax = plt.subplots()
    im = ax.imshow(best_value)

    for i in range(best_value.shape[0]):
        for j in range(best_value.shape[1]):
            if MAP[i][j] in 'GH':
                arrow = MAP[i][j]
            elif best_policy[i, j] == 0:
                arrow = '<'
            elif best_policy[i, j] == 1:
                arrow = 'v'
            elif best_policy[i, j] == 2:
                arrow = '>'
            elif best_policy[i, j] == 3:
                arrow = '^'
            if MAP[i][j] in 'S':
                arrow = 'S ' + arrow
            text = ax.text(j, i, arrow,
                           ha = "center", va = "center", color = "black")

    cbar = ax.figure.colorbar(im, ax = ax)

    fig.tight_layout()
    plt.savefig(figname)

simulator = FrozenLakeEnv(desc = MAP[0])

class agent():
    def __init__(self, epsilon, learning_rate, learning_episodes, map_size,
                 test_interval = 100, action_space = 4, beta = 0.999, do_test = True):
        self.beta = beta
        self.epsilon = epsilon
        self.test_interval = test_interval
        self.batch_num = learning_episodes // test_interval
        self.action_space = action_space
        self.state_space = map_size * map_size + 1
        self.q_table = np.zeros((self.state_space, self.action_space))
        self.learning_rate = learning_rate
        self.do_test = do_test

    def explore_or_exploit_policy(self, curr_state):
        #INSERT YOUR CODE HERE
        action_probs = np.ones(self.action_space) * (self.epsilon/float(self.action_space))
        best_action = np.argmax(self.q_table[curr_state])
        action_probs[best_action] += (1 - self.epsilon)
        action = np.random.choice(np.arange(self.action_space), p=action_probs)
        return action

    def greedy_policy(self, curr_state):
        #INSERT YOUR CODE HERE
        action_probs = np.zeros(self.action_space)
        best_action = np.argmax(self.q_table[curr_state])
        action_probs[best_action] = 1
        action = np.random.choice(np.arange(self.action_space), p=action_probs)
        return action

    def learn_and_evaluate(self):
        total_rewards = []
        for i in tqdm.tqdm(range(self.batch_num), desc="Test Number"):
            #INSERT YOUR CODE HERE
            self.learn()
            total_reward = self.evaluate(self.greedy_policy)
            total_rewards.append(total_reward)

        return total_rewards, self.q_table

    def evaluate(self, policy_func, trials = 100, max_steps = 1000):

        total_reward = 0
        for _ in range(trials):
            simulator.reset()
            done = False
            steps = 0
            observation, reward, done, info = simulator.step(policy_func(0))
            total_reward += (self.beta ** steps) * reward
            steps += 1
            while not done and steps < 1000:
                observation, reward, done, info = simulator.step(policy_func(observation))
                total_reward += (self.beta ** steps) * reward
                steps += 1

        return total_reward / trials

    def learn(self):
        pass


ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

@ray.remote
class QLAgent_server(agent):
    def __init__(self, epsilon, learning_rate, learning_episodes, map_size,
                 test_interval = 100, batch_size = 100, action_space = 4, beta = 0.999, do_test = True):
        print("calling super init")
        super().__init__(epsilon, learning_rate, learning_episodes, map_size,
                         test_interval = test_interval, action_space = action_space, beta = beta, do_test = do_test)
        print("done super init")
        self.collector_done = False
        self.evaluator_done = False
        self.learning_episodes = learning_episodes
        self.episode = 0
        self.reuslts = []
        self.batch_size = batch_size
        self.privous_q_tables = []
        self.results = [0] * (self.batch_num + 1)
        self.reuslt_count = 0
        print("server initialized")

    def learn(self, experiences):
        #INSERT YOUR CODE HERE
        for state, action, reward, next_state in experiences:
            self.q_table[state][action] += learning_rate * (reward + self.beta*np.max(self.q_table[next_state]) - self.q_table[state][action])
        self.episode += 1
        #print("self.episode = ", self.episode)

        if self.do_test:
            if self.episode // self.test_interval + 1 > len(self.privous_q_tables):
                self.privous_q_tables.append(self.q_table)
        return self.get_q_table()

    def get_q_table(self):
        if self.episode >= self.learning_episodes:
            self.collector_done = True

        return self.q_table, self.collector_done

    # evalutor
    def add_result(self, result, num):
        self.results[num] = result

    def get_reuslts(self):
        return self.results, self.q_table

    def ask_evaluation(self):
        #print("@@@@ len(self.privous_q_tables): ", len(self.privous_q_tables))
        if len(self.privous_q_tables) > self.reuslt_count:
            num = self.reuslt_count
            evluation_q_table = self.privous_q_tables[num]
            self.reuslt_count += 1
            return evluation_q_table, False, num
        else:
            if self.episode >= self.learning_episodes:
                self.evaluator_done = True
            return [], self.evaluator_done, None

@ray.remote
def collecting_worker(server, simulator, epsilon, action_space = 4, batch_size = 100):
    def greedy_policy(curr_state):
        #INSERT YOUR CODE HERE
        action_probs = np.zeros(action_space)
        q_table, _ = ray.get(server.get_q_table.remote())
        best_action = np.argmax(q_table[curr_state])
        action_probs[best_action] = 1
        action = np.random.choice(np.arange(action_space), p=action_probs)
        return action

    def explore_or_exploit_policy(curr_state):
        #INSERT YOUR CODE HERE
        action_probs = np.ones(action_space) * (epsilon/float(action_space))
        q_table, _ = ray.get(server.get_q_table.remote())
        best_action = np.argmax(q_table[curr_state])
        action_probs[best_action] += (1 - epsilon)
        action = np.random.choice(np.arange(action_space), p=action_probs)
        return action

    q_table, learn_done = ray.get(server.get_q_table.remote())
    while True:
        if learn_done:
            #print("a collector is done")
            break
        #INSERT YOUR CODE HERE
        simulator.reset()
        episode = []
        done = False
        steps = 0
        state = 0
        action = explore_or_exploit_policy(state)
        next_state, reward, done, info = simulator.step(action)
        episode.append((state, action, reward, next_state))
        state = next_state
        while not done and steps < 1000:
            action = explore_or_exploit_policy(state)
            next_state, reward, done, info = simulator.step(action)
            episode.append((state, action, reward, next_state))
            state = next_state
        q_table, learn_done = ray.get(server.learn.remote(episode))

@ray.remote
def evaluation_worker(server, trials = 100, action_space = 4, beta = 0.999):
    def greedy_policy(curr_state):
        #INSERT YOUR CODE HERE
        action_probs = np.zeros(action_space)
        q_table, _ = ray.get(server.get_q_table.remote())
        best_action = np.argmax(q_table[curr_state])
        action_probs[best_action] = 1
        action = np.random.choice(np.arange(action_space), p=action_probs)
        return action

    while True:
        q_table, done, num = ray.get(server.ask_evaluation.remote())
        #print("----- evaluator checking: done=", done, " num=", num)
        if done:
            #print("$$$$$$$$$$$$ evaluator done")
            break
        if len(q_table) == 0:
            continue
        total_reward = 0
        for _ in range(trials):
        #for _ in range(10):
            simulator.reset()
            done = False
            steps = 0
            observation, reward, done, info = simulator.step(greedy_policy(0))
            total_reward += (beta ** steps) * reward
            steps += 1
            while not done:
                observation, reward, done, info = simulator.step(greedy_policy(observation))
                total_reward += (beta ** steps) * reward
                steps += 1
        server.add_result.remote(total_reward / trials, num)
    #print("----- evaluator done 0")

class distributed_QL_agent():
    def __init__(self, epsilon, learning_rate, learning_episodes, map_size,
                 cw_num = 4, ew_num = 4, test_interval = 100, batch_size = 100,
                 action_space = 4, beta = 0.999, do_test = True):

        self.server = QLAgent_server.remote(epsilon, learning_rate, learning_episodes, map_size,
                                               test_interval = test_interval, batch_size = batch_size, action_space = action_space, do_test = do_test)
        #print("self.server = ", self.server)
        self.workers_id = []
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.agent_name = "Distributed Q-learning"
        self.do_test = do_test
        print("agent initialized")

    def learn_and_evaluate(self):
        workers_id = []

        #INSERT YOUR CODE HERE

        #wait for server to get initialized
        #ray.wait(self.server)

        #initiate all evaluators
        for i in range(self.ew_num):
            w_id = evaluation_worker.remote(self.server)
            workers_id.append(w_id)
        print("evaluators initiated")

        #initiate all collectors
        for i in range(self.cw_num):
            w_id = collecting_worker.remote(self.server, simulator, self.epsilon, action_space = 4, batch_size = self.batch_size)
            workers_id.append(w_id)
        print("collectors initiated")

        print("waiting for everyone to finish")
        ray.wait(workers_id, len(workers_id))
        print("everyone finished")
        return ray.get(self.server.get_reuslts.remote())

simulator.reset()
#INSERT YOUR CODE FOR INIT PARAMS HERE
epsilon = 0.1
#learning_rate = 0.1
learning_rate = 0.001
learning_episodes = 30000 #dangerous hall
#learning_episodes = 100000 #size 16 map
test_interval = 100
#batch_size = learning_episodes // test_interval
batch_size = 100
do_test = True

start_time = time.time()
distributed_ql_agent = distributed_QL_agent(epsilon, learning_rate, learning_episodes, map_size,
                                            test_interval = test_interval, batch_size = batch_size,
                                            cw_num = 8, ew_num = 4, do_test = do_test)
total_rewards, q_table = distributed_ql_agent.learn_and_evaluate()
run_time['Distributed Q-learning agent'] = time.time() - start_time
print("Learning time:\n")
print(run_time['Distributed Q-learning agent'])
if do_test:
    plot_result(total_rewards, test_interval, [distributed_ql_agent.agent_name], 'distributed_ql_agent_performance_curve.png')
plot_image(q_table, MAP[0], 'distributed_ql_agent_qtable.png')
