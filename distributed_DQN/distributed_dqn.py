
import gym
import torch
import time
import os
import ray
import numpy as np
import random
import datetime

from tqdm import tqdm
from random import uniform, randint

import io
import base64
#from IPython.display import HTML

from dqn_model import DQNModel
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt

FloatTensor = torch.FloatTensor

from memory_remote import ReplayBuffer_remote
from dqn_model import _DQNModel
import torch
from custom_cartpole import CartPoleEnv

# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}

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
    plt.savefig("plot")

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


ray.shutdown()
ray.init(include_webui=False, ignore_reinit_error=True, redis_max_memory=500000000, object_store_memory=5000000000)

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole_distributed'

# Register the environment
env_CartPole = CartPoleEnv()

# Set result saveing floder
result_floder = ENV_NAME + "_distributed"
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)
torch.set_num_threads(12)

global_steps = 0

@ray.remote
class DQN_agent(object):
    def __init__(self, env, model_server, memory_server, hyper_params, action_space = len(ACTION_DICT)):

        self.env = env
        self.max_episode_steps = env._max_episode_steps
        """
            (epsilon): The explore or exploit policy epsilon.
            initial_epsilon: When the 'steps' is 0, the epsilon is initial_epsilon, 1
            final_epsilon: After the number of 'steps' reach 'epsilon_decay_steps',
                The epsilon set to the 'final_epsilon' determinately.
            epsilon_decay_steps: The epsilon will decrease linearly along with the steps from 0 to 'epsilon_decay_steps'.
        """
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        """
            episode: Record training episode
            steps: Add 1 when predicting an action
            action_space: The action space of the current environment, e.g 2.
        """
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.action_space = action_space
        """
            batch_size: Mini batch size for training model.
            update_steps: The frequence of traning model
            model_replace_freq: The frequence of replacing 'target_model' by 'eval_model'
        """
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']

        self.memory_server = memory_server
        self.model_server = model_server

    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, final_decay_steps):
        global global_steps
        decay_rate = global_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(self, state):
        p = uniform(0, 1)
        # Get decreased epsilon
        epsilon = self.linear_decrease(self.initial_epsilon,
                               self.final_epsilon,
                               self.epsilon_decay_steps)

        if p < epsilon:
            #return action
            return randint(0, self.action_space - 1)
        else:
            #return action
            return ray.get(self.model_server.greedy_policy.remote(state))

    def learn(self, test_interval):
        global global_steps
        for episode in range(test_interval):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                #INSERT YOUR CODE HERE
                # add experience from explore-exploit policy to memory
                # update the model every 'update_steps' of experience
                # update the target network (if the target network is being used) every 'model_replace_freq' of experiences
                action = self.explore_or_exploit_policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory_server.add.remote(state, action, reward, next_state, done)
                state = next_state

                if self.steps % self.update_steps == 0:
                    self.model_server.update_batch.remote(self.batch_size)
                if steps % self.model_replace_freq == 0:
                    self.model_server.replace_target.remote()
                steps += 1
                global_steps +=1
                #print (global_steps)
@ray.remote
class Model_Server(object):
    def __init__(self, env, hyper_params, memory_server):
        """
            input_len The input length of the neural network. It equals to the length of the state vector.
            output_len: The output length of the neural network. It is equal to the action space.
            eval_model: The model for predicting action for the agent.
            target_model: The model for calculating Q-value of next_state to update 'eval_model'.
            use_target_model: Trigger for turn 'target_model' on/off
        """
        self.beta = hyper_params['beta']

        state = env.reset()
        action_space = len(ACTION_DICT)
        input_len = len(state)
        output_len = action_space
        self.eval_model = DQNModel(input_len, output_len, learning_rate = hyper_params['learning_rate'])
        self.use_target_model = hyper_params['use_target_model']
        if self.use_target_model:
            self.target_model = DQNModel(input_len, output_len)

        self.memory_server = memory_server

    def update_batch(self, batch_size):

        batch = ray.get(self.memory_server.sample.remote(batch_size))

        (states, actions, reward, next_states,
         is_terminal) = batch

        states = states
        next_states = next_states
        terminal = FloatTensor([0 if t else 1 for t in is_terminal])
        reward = FloatTensor(reward)
        batch_index = torch.arange(batch_size,
                                   dtype=torch.long)

        # Current Q Values
        _, q_values = self.eval_model.predict_batch(states)
        q_values = q_values[batch_index, actions]

        # Calculate target
        if self.use_target_model:
            actions, q_next = self.target_model.predict_batch(next_states)
            q_next = q_next[batch_index, actions]
        else:
            actions, q_next = self.eval_model.predict_batch(next_states)
            q_next = q_next[batch_index, actions]

        #INSERT YOUR CODE HERE --- neet to compute 'q_targets' used below
        q_max = q_next * terminal
        q_target = reward + self.beta * q_max

        # update model
        self.eval_model.fit(q_values, q_target)

    def replace_target(self):
        self.target_model.replace(self.eval_model)

    def greedy_policy(self, state):
        return self.eval_model.predict(state)

@ray.remote
class DQN_Evaluator(object):
    def __init__(self, env, hyper_params, model_server):
        self.env = env
        state = self.env.reset()
        self.best_reward = 0
        self.max_episode_steps = env._max_episode_steps

        self.model_server = model_server

    def evaluate(self, trials = 30):
        total_reward = 0
        for _ in range(trials):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                steps += 1
                action = ray.get(self.model_server.greedy_policy.remote(state))
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        print(avg_reward)

        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward
        return avg_reward

    # save model
    def save_model(self):
        self.eval_model.save(result_floder + '/best_model.pt')

    # load model
    def load_model(self):
        self.eval_model.load(result_floder + '/best_model.pt')

training_episodes, test_interval = 15000, 50

test_number = training_episodes // test_interval
all_results = []


# Create a memory server
memory_server = ReplayBuffer_remote.remote(hyperparams_CartPole['memory_size'])

# Create a model server
model_server = Model_Server.remote(env_CartPole, hyperparams_CartPole, memory_server)

# Create the collectors and Evaluators numbers
num_collectors = 4
num_evaluators = 4

collectors_workers = []
evaluators = []

a = datetime.datetime.now()

for i in range(num_collectors):
    env_CartPole = CartPoleEnv()
    collectors_workers.append(DQN_agent.remote(env_CartPole, model_server, memory_server, hyperparams_CartPole))
for j in range(num_evaluators):
    env_CartPole = CartPoleEnv()
    evaluators.append(DQN_Evaluator.remote(env_CartPole, hyperparams_CartPole, model_server))

# start the all worker, store their id in a list
workers_id = []
evaluators_result = []

for _ in range(test_number):
    select_server = random.randint(0, num_collectors - 1)
    evaluator = random.randint(0, num_evaluators - 1 )

    collection_worker_id = collectors_workers[select_server].learn.remote(test_interval)
    workers_id.append( collection_worker_id )
    evaluation_worker_id = evaluators[evaluator].evaluate.remote()
    evaluators_result.append( evaluation_worker_id )

ray.wait(workers_id, len(workers_id))
ray.wait(evaluators_result, len(evaluators_result))

b = datetime.datetime.now()

result = (ray.get(evaluators_result))

plot_result(result, test_interval, ["batch_update with target_model"])

c = b - a

print (c.total_seconds())
