from . import RLAgent
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os

class DQNAgent(RLAgent):
    def __init__(self, action_space, ob_generator, reward_generator, iid):
        super().__init__(action_space, ob_generator, reward_generator)

        self.iid = iid

        self.ob_length = ob_generator.ob_length

        self.memory = deque(maxlen=2000)
        self.learning_start = 2000
        self.update_model_freq = 10
        self.update_target_model_freq = 100

        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.batch_size = 32

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def get_action(self, ob):
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
        ob = self._reshape_ob(ob)
        act_values = self.model.predict(ob)
        return np.argmax(act_values[0])

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(40, input_dim=self.ob_length, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(
            loss='mse',
            optimizer=Adam(lr=self.learning_rate)
        )
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        weights = self.model.get_weights()
        self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        self.memory.append((ob, action, reward, next_ob))

    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)
        for ob, action, reward, next_ob in minibatch:
            # print(next_state)
            ob = self._reshape_ob(ob)
            next_ob = self._reshape_ob(next_ob)
            target = (reward + self.gamma *
                      np.amax(self.target_model.predict(next_ob)[0]))
            target_f = self.model.predict(ob)
            target_f[0][action] = target
            history = self.model.fit(ob, target_f, epochs=1, verbose=0)
        # print(history.history['loss'])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn"):
        name = "dqn_agent_{}.h5".format(self.iid)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)