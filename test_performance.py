from torch_agent import MCTS_Torch, ConnectFour, ResNet_Torch, TorchAgent
from tf_agent import ResNet_tf, TFAgent
from tf_agent_tflite import TFLightAgent
import torch
import matplotlib.pyplot as plt
import json


# Initialize the ConnectFour game
game = ConnectFour()

### INITIALIZE PARAMETER FOR THE FIRST AGENT ###

args_torch = {
    'C': 2,
    'num_searches': 100,
    'dirichlet_epsilon': 0.,
    'dirichlet_alpha': 0.3
}
# Set the device to GPU if available, otherwise CPU
device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model_torch = ResNet_Torch(game, 9, 128, device_torch)
model_torch.load_state_dict(torch.load("model_7_ConnectFour.pt", map_location=device_torch))
model_torch.eval()

# Initialize the TorchAgent with the model and args
torch_agent = TorchAgent(model_torch, args_torch)

### INITIALIZE PARAMETER FOR THE SECOND AGENT ###


# Initialize the TensowFlow Agent with the choosen model and args
args_tf = {
    'C': 2,
    'num_searches': 100,
    'dirichlet_epsilon': 0.0, 
    'dirichlet_alpha': 0.3
}

tf_agent = TFLightAgent(args_tf)


def start_testing(num_games=10, save_results=True):
    """
    Function to start testing different Agents in the context of playing connect four.
    
    Args:
        num_games (int): Number of games to play.
        save_results (bool): Whether to save the results to a JSON file.
    """

    global player, mcts
    # Get the initial state of the game
    results = []
    player1_wins = 0
    player2_wins = 0
    draws = 0

    for game_idx in range(1, num_games + 1):
        
        # Change starting player for each game
        player = 1 if game_idx % 2 == 1 else -1 
        state = game.get_initial_state()


        while True:
            
            if player == 1:
                action = torch_agent.getAction(state)
            else:
                neutral_state = game.change_perspective(state, player)
                action = tf_agent.getAction(neutral_state)
                
            state = game.get_next_state(state, action, player)
            value, is_terminal = game.get_value_and_terminated(state, action)

            if is_terminal:
                print(f"Game {game_idx} ended.")
                print(state)

                if value == 1:
                    print(f"Player {player} won!")
                    if player == 1:
                        player1_wins += 1
                        results.append("Player 1 win")
                    else:
                        player2_wins += 1
                        results.append("Player 2 win")
                else:
                    print("Game ended in a draw.")
                    draws += 1
                    results.append("Draw")
                break

            player = game.get_opponent(player)
    
    # Summary
    print("\n--- Testing Summary ---")
    print(f"Total games: {num_games}")
    print(f"Player 1 wins: {player1_wins} ({player1_wins/num_games:.1%})")
    print(f"Player 2 wins: {player2_wins} ({player2_wins/num_games:.1%})")
    print(f"Draws: {draws} ({draws/num_games:.1%})")

    # Plot
    game_numbers = list(range(1, num_games + 1))
    outcomes = [1 if r == "Player 1 win" else -1 if r == "Player 2 win" else 0 for r in results]

    plt.figure(figsize=(10, 5))
    plt.plot(game_numbers, outcomes, marker='o', linestyle='-', label="Game Outcome")
    plt.yticks([-1, 0, 1], ["Player 2 Win", "Draw", "Player 1 Win"])
    plt.xlabel("Game Number")
    plt.ylabel("Outcome")
    plt.title("Game Outcomes Over Time")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Save results
    if save_results:
        summary_data = {
            "total_games": num_games,
            "player1_wins": player1_wins,
            "player2_wins": player2_wins,
            "draws": draws,
            "results": results
        }
        with open("testing_results.json", "w") as f:
            json.dump(summary_data, f, indent=4)
        print("Results saved to testing_results.json.")


# Start testing with 100 games and save results
start_testing(10, True)