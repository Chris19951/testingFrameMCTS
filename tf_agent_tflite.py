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

        return encoded_state




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


def _prep_tflite_input(encoded_state, input_details):
    """
    Wandelt ein (N, C, H, W)‑Array in das vom TFLite‑Interpreter erwartete
    Format um ‑ inkl. optionaler Quantisierung.
    """
    # TFLite erwartet NHWC
    if encoded_state.shape[1] in (1, 3):               # (N,C,H,W) ➜ (N,H,W,C)
        input_data = np.transpose(encoded_state, (0, 2, 3, 1))
    else:                                              # schon NHWC
        input_data = encoded_state

    scale, zero_pt = input_details[0]['quantization']
    if scale > 0:                                      # quantisiertes Modell?
        input_data = np.round(input_data / scale + zero_pt)
        input_data = np.clip(input_data, -128, 127).astype(np.int8)
    else:
        input_data = input_data.astype(np.float32)

    return input_data


def _dequant_tflite_output(tensor, details):
    scale, zero_pt = details['quantization']
    if scale > 0:
        return scale * (tensor.astype(np.float32) - zero_pt)
    return tensor

def load_tflite_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    return interpreter

def _softmax_numpy(x, axis=-1):
    """
    stabile Softmax für NumPy‑Arrays
    """
    x = x - np.max(x, axis=axis, keepdims=True)  # log‑trick
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)



class MCTS:
    def __init__(self, game, args):
        self.game = game
        self.args = args

    #@tf.function  # CHANGE: Graph-Mode für schnellere Inferenz
    def predict(self, encoded_state, interpreter):
        #print(encoded_state)
        # ------- Input vorbereiten
        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_data     = _prep_tflite_input(encoded_state, input_details)

        interpreter.set_tensor(input_details[0]['index'], input_data)

        # ------- Inferenz
        #start = time.perf_counter()
        interpreter.invoke()
        #dur_us = (time.perf_counter() - start) * 1_000_000
        #print(f"TFLite inference time: {dur_us:.0f} µs")

        # ------- Outputs holen & ggf. de‑quantisieren
        policy_logits = _dequant_tflite_output(
            interpreter.get_tensor(output_details[1]['index']),
            output_details[1]
        ).flatten()

        value = _dequant_tflite_output(
            interpreter.get_tensor(output_details[0]['index']),
            output_details[0]
        ).flatten()

        policy = _softmax_numpy(policy_logits)

        return policy, float(value.squeeze())



    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)
        interpreter = load_tflite_model("raspi_model_connectFour_integer_quant.tflite")
        encoded_state = np.array(self.game.get_encoded_state(state))[np.newaxis, ...].astype(np.float32)
        #print(f"Encoded state: {encoded_state}")
        policy, _ = self.predict(encoded_state, interpreter)  # CHANGE: predict mit tf.function
        #policy = tf.nn.softmax(policy_logits, axis=1).numpy().squeeze(0)

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
                encoded_state = np.array(self.game.get_encoded_state(node.state))[np.newaxis, ...].astype(np.float32)
                policy, value_out = self.predict(encoded_state, interpreter)  # CHANGE: predict mit tf.function


                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                if np.sum(policy) > 0:
                    policy /= np.sum(policy)
                else:
                    policy = valid_moves / np.sum(valid_moves)

                value = value_out

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
                



class TFLightAgent:
    def __init__(self, args):
        self.game = ConnectFour()
        self.args = args
        self.mcts = MCTS(self.game, args)

    def getAction(self, state):
        mcts_probs = self.mcts.search(state)
        action = np.argmax(mcts_probs)

        return action

"""# Start the training process
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
alphaZero.learn()"""

