import random
import time
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class ConnectFour:
    def __init__(self):
        self.row_count = 6
        self.column_count = 7
        self.action_size = self.column_count
        self.in_a_row = 4
        
    def __repr__(self):
        return "ConnectFour"
        
    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = np.max(np.where(state[:, action] == 0))
        state[row, action] = player
        return state
    
    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        if action is None:
            return False
        
        row = np.min(np.where(state[:, action] != 0))
        column = action
        player = state[row][column]

        def count(offset_row, offset_column):
            for i in range(1, self.in_a_row):
                r = row + offset_row * i
                c = action + offset_column * i
                if (
                    r < 0 
                    or r >= self.row_count
                    or c < 0 
                    or c >= self.column_count
                    or state[r][c] != player
                ):
                    return i - 1
            return self.in_a_row - 1

        return (
            count(1, 0) >= self.in_a_row - 1
            or (count(0, 1) + count(0, -1)) >= self.in_a_row - 1
            or (count(1, 1) + count(-1, -1)) >= self.in_a_row - 1
            or (count(1, -1) + count(-1, 1)) >= self.in_a_row - 1
        )

    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player):
        return -player

    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        encoded_state = np.transpose(encoded_state, (1, 2, 0))
        encoded_state = np.expand_dims(encoded_state, axis=0)
        return tf.convert_to_tensor(encoded_state, dtype=tf.float32)  # CHANGE: Direkt Tensor zurückgeben


class ResNet_tf(tf.keras.Model):
    def __init__(self, game, num_resBlocks, num_hidden):
        super().__init__()
        
        self.startBlock = keras.Sequential([
            layers.Conv2D(num_hidden, 3, padding="same", input_shape=(game.row_count, game.column_count, 3)),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])
        
        self.backBone = [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        
        self.policyHead = keras.Sequential([
            layers.Conv2D(32, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(game.action_size)
        ])
        
        self.valueHead = keras.Sequential([
            layers.Conv2D(3, 3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Flatten(),
            layers.Dense(1, activation="tanh")
        ])

    def call(self, x, training=False):
        x = self.startBlock(x, training=training)
        for resBlock in self.backBone:
            x = resBlock(x, training=training)
        policy = self.policyHead(x, training=training)
        value = self.valueHead(x, training=training)
        return policy, value
        
class ResBlock(tf.keras.Model):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = layers.Conv2D(num_hidden, 3, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(num_hidden, 3, padding="same")
        self.bn2 = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
    def call(self, x, training=False):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x += residual
        return self.relu(x)
    

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_fully_expanded(self):
        return len(self.children) > 0
    
    def select(self):
        best_child = None
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
                
        return best_child
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)
                
        return child
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)  


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @tf.function  # CHANGE: Graph-Mode für schnellere Inferenz
    def predict(self, x):
        return self.model(x, training=False)

    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        
        encoded_state = self.game.get_encoded_state(state)  # Tensor direkt
        policy_logits, _ = self.predict(encoded_state)  # CHANGE: predict mit tf.function
        policy = tf.nn.softmax(policy_logits, axis=1).numpy().squeeze(0)

        dirichlet_noise = np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * dirichlet_noise
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        if np.sum(policy) > 0:
            policy /= np.sum(policy)
        else:
            policy = valid_moves / np.sum(valid_moves)

        root.expand(policy)
        
        for _ in range(self.args['num_searches']):
            node = root
            while node.is_fully_expanded():
                node = node.select()
                
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)
            value = self.game.get_opponent_value(value)
            
            if not is_terminal:
                encoded_state = self.game.get_encoded_state(node.state)  # Tensor direkt
                policy_logits, value_out = self.predict(encoded_state)  # CHANGE: predict mit tf.function
                policy = tf.nn.softmax(policy_logits, axis=1).numpy().squeeze(0)

                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                if np.sum(policy) > 0:
                    policy /= np.sum(policy)
                else:
                    policy = valid_moves / np.sum(valid_moves)

                value = value_out.numpy().item()

                node.expand(policy)
                
            node.backpropagate(value)    
        
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

    

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        
        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)
            
            memory.append((neutral_state, action_probs, player))
            
            action = np.random.choice(self.game.action_size, p=action_probs)
            state = self.game.get_next_state(state, action, player)
            
            value, is_terminal = self.game.get_value_and_terminated(state, action)
            
            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    returnMemory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return returnMemory
            
            player = self.game.get_opponent(player)
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory), batchIdx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            state = tf.concat(state, axis=0)  # alle Tensoren zusammenführen
            policy_targets = np.array(policy_targets)
            value_targets = np.array(value_targets).reshape(-1, 1)
            
            with tf.GradientTape() as tape:
                out_policy, out_value = self.model(state, training=True)
                policy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=policy_targets, logits=out_policy))
                value_loss = tf.reduce_mean(tf.square(out_value - value_targets))
                loss = policy_loss + value_loss
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            print(f"[Iteration {iteration + 1}/{self.args['num_iterations']}]")
            memory = []

            for selfPlay_iteration in range(self.args['num_selfPlay_iterations']):
                print(f"  Self-Play Iteration {selfPlay_iteration + 1}/{self.args['num_selfPlay_iterations']}")
                memory += self.selfPlay()

            for epoch in range(self.args['num_epochs']):
                print(f"  Training Epoch {epoch + 1}/{self.args['num_epochs']}")
                self.train(memory)

            self.model.save_weights(f"model_{iteration}_{self.game}.weights.h5")


class TFAgent:
    def __init__(self, model, args):
        self.game = ConnectFour()
        self.model = model
        self.args = args
        self.mcts = MCTS(self.game, args, model)

    def getAction(self, state):
        mcts_probs = self.mcts.search(state)
        action = np.argmax(mcts_probs)

        return action

# Start the training process
game = ConnectFour()

model = ResNet_tf(game, num_resBlocks=9, num_hidden=128)
model.build(input_shape=(None, game.row_count, game.column_count, 3))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 2,
    'num_selfPlay_iterations': 500,
    'num_epochs': 4,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.3
}

alphaZero = AlphaZero(model, optimizer, game, args)
alphaZero.learn()

