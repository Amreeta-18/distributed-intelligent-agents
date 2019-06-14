#import gym
import time
import os
import ray
import numpy as np
import time

from tqdm import tqdm
from random import uniform, randint

import io
import base64
#from IPython.display import HTML

from dqn_model import DQNModel

from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv

import matplotlib.pyplot as plt

FloatTensor = torch.FloatTensor

#initialize ray
ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Set result saveing floder
result_floder = ENV_NAME
result_file = result_floder + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)

# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}

# make environment
env_CartPole = CartPoleEnv()


def plot_result(total_rewards ,learning_num, legend):
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
    plt.savefig('curve.png')


hyperparams_CartPole = {
    'epsilon_decay_steps' : 100000,
    'final_epsilon' : 0.1,
    'batch_size' : 32,
    'update_steps' : 10,
    'memory_size' : 2000,
    'beta' : 0.99,
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}


#@ray.remote
class memory_server():
    def __init__(self, hyper_params):
        self.memory = ReplayBuffer_remote.remote(hyper_params['memory_size'])

    def add(self, obs_t, action, reward, obs_tp1, done):
        self.memory.add.remote(obs_t, action, reward, obs_tp1, done)

    def get_current_size(self):
        return ray.get(self.memory.__len__.remote())

    def sample(self, batch_size):
        return ray.get(self.memory.sample.remote(batch_size))


@ray.remote
class DQN_collector():
    def __init__(self, env, hyper_params, action_space):
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']
        self.steps = 0
        self.best_reward = 0
        self.action_space = action_space
        self.use_target_model = hyper_params['use_target_model']

    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(self, state, server):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon,
                               self.final_epsilon,
                               self.steps,
                               self.epsilon_decay_steps)
        #if(np.random.randint(1000)==4):
            #print("epsilon",epsilon)
        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return self.greedy_policy(state, server)

    def greedy_policy(self, state, server):
        return ray.get(server.eval_model_predict.remote(state))

    def learn(self, test_interval, memory, server):
        for episode in tqdm(range(test_interval), desc="Training"):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                #INSERT YOUR CODE HERE
                # add experience from explore-exploit policy to memory
                action = self.explore_or_exploit_policy(state, server)
                next_state, reward, done, info = self.env.step(action)
                memory.add(state, action, reward, next_state, done)

                # update the model every 'update_steps' of experience
                server.update_batch.remote(memory)

                # update the target network (if the target network is being used) every 'model_replace_freq' of experiences
                if self.use_target_model and (self.steps % self.model_replace_freq == 0):
                    server.replace_target_model.remote()

                self.steps += 1
                steps += 1
                state = next_state



@ray.remote
class DQN_server():
    def __init__(self, env, hyper_params, action_space):

        #self.env = env
        #self.max_episode_steps = env._max_episode_steps

        """
            beta: The discounted factor of Q-value function
            (epsilon): The explore or exploit policy epsilon.
            initial_epsilon: When the 'steps' is 0, the epsilon is initial_epsilon, 1
            final_epsilon: After the number of 'steps' reach 'epsilon_decay_steps',
                The epsilon set to the 'final_epsilon' determinately.
            epsilon_decay_steps: The epsilon will decrease linearly along with the steps from 0 to 'epsilon_decay_steps'.
        """
        self.beta = hyper_params['beta']

        """
            episode: Record training episode
            steps: Add 1 when predicting an action
            learning: The trigger of agent learning. It is on while training agent. It is off while testing agent.
            action_space: The action space of the current environment, e.g 2.
        """
        # self.episode = 0
        # self.steps = 0
        # self.best_reward = 0
        # self.learning = True
        # self.action_space = action_space

        """
            input_len The input length of the neural network. It equals to the length of the state vector.
            output_len: The output length of the neural network. It is equal to the action space.
            eval_model: The model for predicting action for the agent.
            target_model: The model for calculating Q-value of next_state to update 'eval_model'.
            use_target_model: Trigger for turn 'target_model' on/off
        """
        state = env.reset()
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)
#         memory: Store and sample experience replay.
        #self.memory = ReplayBuffer(hyper_params['memory_size'])

        """
            batch_size: Mini batch size for training model.
            update_steps: The frequence of traning model
            model_replace_freq: The frequence of replacing 'target_model' by 'eval_model'
        """
        self.batch_size = hyper_params['batch_size']
        #self.update_steps = hyper_params['update_steps']
        #self.model_replace_freq = hyper_params['model_replace_freq']

        print("server initialized")

    def replace_target_model(self):
        self.target_model.replace(self.eval_model)

    def eval_model_predict(self, state):
        return self.eval_model.predict(state)

    # This next function will be called in the main RL loop to update the neural network model given a batch of experience
    # 1) Sample a 'batch_size' batch of experiences from the memory.
    # 2) Predict the Q-value from the 'eval_model' based on (states, actions)
    # 3) Predict the Q-value from the 'target_model' base on (next_states), and take the max of each Q-value vector, Q_max
    # 4) If is_terminal == 1, q_target = reward + discounted factor * Q_max, otherwise, q_target = reward
    # 5) Call fit() to do the back-propagation for 'eval_model'.
    def update_batch(self, memory):
        current_memory_size = memory.get_current_size()
        if current_memory_size < self.batch_size:
            return

        #print("fetching minibatch from replay memory")
        batch = memory.sample(self.batch_size)

        (states, actions, reward, next_states,
         is_terminal) = batch

        states = states
        next_states = next_states
        terminal = FloatTensor([1 if t else 0 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(self.batch_size,
                                   dtype=torch.long)

        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)

        #q_values = q_values[np.arange(self.batch_size), actions]
        q_values = q_values[batch_index, actions]

        # Calculate target
        if self.use_target_model:
            #print("target_model.predict")
            best_actions, q_next = self.target_model.predict_batch(next_states)
        else:
            best_actions, q_next = self.eval_model.predict_batch(next_states)

        q_max = q_next[batch_index, best_actions]

        terminal = 1 - terminal
        q_max *= terminal
        q_target = reward + self.beta * q_max

        # update model
        self.eval_model.fit(q_values, q_target)

    # save model
    def save_model(self):
        self.eval_model.save(result_floder + '/best_model.pt')

    # load model
    def load_model(self):
        self.eval_model.load(result_floder + '/best_model.pt')


@ray.remote
class DQN_evaluator():
    def __init__(self, env):
        self.env = env
        self.max_episode_steps = env._max_episode_steps
        self.best_reward = 0

    def greedy_policy(self, state, server):
        return ray.get(server.eval_model_predict.remote(state))

    def evaluate(self, server, trials = 30):
        total_reward = 0
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                steps += 1
                action = self.greedy_policy(state, server)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        print(avg_reward)
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward
            server.save_model.remote()
        return avg_reward


class distributed_DQN_agent():
    def __init__(self, env, hyper_params, action_space, cw_num, ew_num):
        self.env = env
        self.hyper_params = hyper_params
        self.action_space = action_space
        self.server = DQN_server.remote(env, hyper_params, action_space)
        self.memory = memory_server(hyper_params)
        self.cw_num = cw_num
        self.ew_num = ew_num
        self.agent_name = "Distributed DQN"
        print("agent initialized")


    def learn_and_evaluate(self, training_episodes, test_interval):
        test_number = training_episodes // test_interval
        all_results = []

        for i in range(test_number):
            workers_id = []

            # learn
            #initiate all collectors
            for j in range(self.cw_num):
                collector = DQN_collector.remote(self.env, self.hyper_params, self.action_space)
                w_id = collector.learn.remote(test_interval, self.memory, self.server)
                workers_id.append(w_id)
            print("collectors initiated")

            #wait for all collectors to finish
            ray.wait(workers_id, len(workers_id))

            workers_id = []

            # evaluate
            #initiate all evaluators
            for j in range(self.ew_num):
                evaluator = DQN_evaluator.remote(self.env)
                w_id  = evaluator.evaluate.remote(self.server)
                workers_id.append(w_id)
            print("evaluators initiated")

            #wait for all evaluators to finish
            ew_ids, _ = ray.wait(workers_id, len(workers_id))
            #get evaluators' results
            evaluator_results = ray.get(ew_ids)

            #get average reward from evaluators' results
            #print("---------------------- len(evaluator_results): ", len(evaluator_results))
            avg_reward = sum(evaluator_results) / float(len(evaluator_results))
            all_results.append(avg_reward)

        return all_results



#debug
training_episodes, test_interval = 2500, 50
num_collectors, num_evaluators = 4, 2
agent = distributed_DQN_agent(env_CartPole, hyperparams_CartPole, len(ACTION_DICT), num_collectors, num_evaluators)
t1 = time.time()
result = agent.learn_and_evaluate(training_episodes, test_interval)
t2 = time.time()
print("TIME TAKEN = ", t2-t1)
plot_result(result, test_interval, ["batch_update with target_model"])
#plot_result(result, test_interval, ["standard_QL_with_FA"])
