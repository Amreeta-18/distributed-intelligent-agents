import gym
import torch
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
from dqn_model import _DQNModel
from memory import ReplayBuffer

import matplotlib.pyplot as plt

FloatTensor = torch.FloatTensor

# Set the Env name and action space for CartPole
ENV_NAME = 'CartPole-v0'
# Move left, Move right
ACTION_DICT = {
    "LEFT": 0,
    "RIGHT":1
}
# Register the environment
env_CartPole = gym.make(ENV_NAME)

# Set result saveing floder
result_floder = ENV_NAME
result_file = ENV_NAME + "/results.txt"
if not os.path.isdir(result_floder):
    os.mkdir(result_floder)

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
    'batch_size' : 1,
    'update_steps' : 1,
    'memory_size' : 1,
    'beta' : 0.99,
    'model_replace_freq' : 2000,
    'learning_rate' : 0.0003,
    'use_target_model': True
}



class DQN_agent(object):
    def __init__(self, env, hyper_params, action_space = len(ACTION_DICT)):

        self.env = env
        self.max_episode_steps = env._max_episode_steps

        """
            beta: The discounted factor of Q-value function
            (epsilon): The explore or exploit policy epsilon.
            initial_epsilon: When the 'steps' is 0, the epsilon is initial_epsilon, 1
            final_epsilon: After the number of 'steps' reach 'epsilon_decay_steps',
                The epsilon set to the 'final_epsilon' determinately.
            epsilon_decay_steps: The epsilon will decrease linearly along with the steps from 0 to 'epsilon_decay_steps'.
        """
        self.beta = hyper_params['beta']
        self.initial_epsilon = 1
        self.final_epsilon = hyper_params['final_epsilon']
        self.epsilon_decay_steps = hyper_params['epsilon_decay_steps']

        """
            episode: Record training episode
            steps: Add 1 when predicting an action
            learning: The trigger of agent learning. It is on while training agent. It is off while testing agent.
            action_space: The action space of the current environment, e.g 2.
        """
        self.episode = 0
        self.steps = 0
        self.best_reward = 0
        self.learning = True
        self.action_space = action_space

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
        self.memory = ReplayBuffer(hyper_params['memory_size'])

        """
            batch_size: Mini batch size for training model.
            update_steps: The frequence of traning model
            model_replace_freq: The frequence of replacing 'target_model' by 'eval_model'
        """
        self.batch_size = hyper_params['batch_size']
        self.update_steps = hyper_params['update_steps']
        self.model_replace_freq = hyper_params['model_replace_freq']

        print("agent initialized")

    # Linear decrease function for epsilon
    def linear_decrease(self, initial_value, final_value, curr_steps, final_decay_steps):
        decay_rate = curr_steps / final_decay_steps
        if decay_rate > 1:
            decay_rate = 1
        return initial_value - (initial_value - final_value) * decay_rate

    def explore_or_exploit_policy(self, state):
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
            return self.greedy_policy(state)

    def greedy_policy(self, state):
        return self.eval_model.predict(state)

    # This next function will be called in the main RL loop to update the neural network model given a batch of experience
    # 1) Sample a 'batch_size' batch of experiences from the memory.
    # 2) Predict the Q-value from the 'eval_model' based on (states, actions)
    # 3) Predict the Q-value from the 'target_model' base on (next_states), and take the max of each Q-value vector, Q_max
    # 4) If is_terminal == 1, q_target = reward + discounted factor * Q_max, otherwise, q_target = reward
    # 5) Call fit() to do the back-propagation for 'eval_model'.
    def update_batch(self):
        if len(self.memory) < self.batch_size or self.steps % self.update_steps != 0:
            return

        #print("fetching minibatch from replay memory")
        batch = self.memory.sample(self.batch_size)

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


    def learn_and_evaluate(self, training_episodes, test_interval):
        test_number = training_episodes // test_interval
        all_results = []

        for i in range(test_number):
            # learn
            self.learn(test_interval)

            # evaluate
            avg_reward = self.evaluate()
            all_results.append(avg_reward)

        return all_results

    def learn(self, test_interval):
        for episode in tqdm(range(test_interval), desc="Training"):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                #INSERT YOUR CODE HERE
                # add experience from explore-exploit policy to memory
                action = self.explore_or_exploit_policy(state)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add(state, action, reward, next_state, done)

                # update the model every 'update_steps' of experience
                self.update_batch()

                # update the target network (if the target network is being used) every 'model_replace_freq' of experiences
                if self.use_target_model and (self.steps % self.model_replace_freq == 0):
                    self.target_model.replace(self.eval_model)

                self.steps += 1
                steps += 1
                state = next_state

    def evaluate(self, trials = 30):
        total_reward = 0
        for _ in tqdm(range(trials), desc="Evaluating"):
            state = self.env.reset()
            done = False
            steps = 0

            while steps < self.max_episode_steps and not done:
                steps += 1
                action = self.greedy_policy(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward

        avg_reward = total_reward / trials
        print(avg_reward)
        f = open(result_file, "a+")
        f.write(str(avg_reward) + "\n")
        f.close()
        if avg_reward >= self.best_reward:
            self.best_reward = avg_reward
            self.save_model()
        return avg_reward

    # save model
    def save_model(self):
        self.eval_model.save(result_floder + '/best_model.pt')

    # load model
    def load_model(self):
        self.eval_model.load(result_floder + '/best_model.pt')


#debug
training_episodes, test_interval = 10000, 50
agent = DQN_agent(env_CartPole, hyperparams_CartPole)
t1 = time.time()
result = agent.learn_and_evaluate(training_episodes, test_interval)
t2 = time.time()
print("TIME TAKEN = ", t2-t1)
#plot_result(result, test_interval, ["batch_update with target_model"])
plot_result(result, test_interval, ["DQN_ReplayBuffer_noTargetModel"])


#run
# training_episodes, test_interval = 10000, 50
# agent = DQN_agent(env_CartPole, hyperparams_CartPole)
# result = agent.learn_and_evaluate(training_episodes, test_interval)
# plot_result(result, test_interval, ["batch_update with target_model"])
