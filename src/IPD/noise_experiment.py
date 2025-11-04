import numpy as np
import matplotlib.pyplot as plt
import os
from .strategies import AlwaysCooperate, AlwaysDefect, TitForTat, RandomStrategy, SuspiciousTitForTat, GrudgerStrategy,TitForTwoTats, GeneticStrategy
from .game import IPDGame
from .genetic_algorithm import IPDGeneticAlgorithm

# Define results directory for saving outputs 
SAVE_PLOTS = True 
PLOT_DIR = "plots"  
if SAVE_PLOTS and not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def run_noise_experiment(noise_levels=[0.0, 0.05, 0.1, 0.2]):
    opponents = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        RandomStrategy(),
        SuspiciousTitForTat(),
        GrudgerStrategy()
    ]
    
    class NoisyIPDGame(IPDGame):
        def __init__(self, rounds=200, noise_level=0.0):
            super().__init__(rounds)
            self.noise_level = noise_level
        
        def play_game(self, player1, player2):
            player1.reset()
            player2.reset()
        
            player1_history = []
            player2_history = []
            player1_score = 0
            player2_score = 0
            
            for _ in range(self.rounds):
    
                player1_move = player1.make_move(player2_history)
                player2_move = player2.make_move(player1_history)
                
                if np.random.random() < self.noise_level:
                    player1_move = 1 - player1_move  
                if np.random.random() < self.noise_level:
                    player2_move = 1 - player2_move  
                
                player1_history.append(player1_move)
                player2_history.append(player2_move)

                round_score = self.play_round(player1_move, player2_move)
                player1_score += round_score[0]
                player2_score += round_score[1]
            
            return player1_score, player2_score
    results = {}
    
    for noise in noise_levels:
        print(f"\nRunning experiment with noise level: {noise}")
        noisy_game = NoisyIPDGame(rounds=200, noise_level=noise)
        
        class NoisyIPDGeneticAlgorithm(IPDGeneticAlgorithm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.game = noisy_game
        
        ga = NoisyIPDGeneticAlgorithm(
            opponents=opponents,
            population_size=100,
            elite_size=10,
            tournament_size=5,
            mutation_rate=0.1,
            crossover_rate=0.7,
            memory_length=1,
            use_probabilities=False
        )

        ga_results = ga.evolve(generations=100)
        results[f"Noise {noise}"] = {
            "best_strategy": ga.best_solution,
            "description": ga.get_strategy_description(),
            "best_fitness": ga.best_fitness,
            "fitness_history": ga.best_fitness_history
        }
        
        print(f"Best strategy: {ga.get_strategy_description()}")
    
    plt.figure(figsize=(12, 8))
    for name, result in results.items():
        plt.plot(result["fitness_history"], label=name)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (Total Score)")
    plt.title("Evolution in noisy environments")
    plt.legend()
    plt.grid(True)
    if SAVE_PLOTS:
        filename = f"{PLOT_DIR}/noise_experiment_fitness_history.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show() 
    plt.close()
    
    print("\nAnalysing robustness of evolved strategies:")
    robustness_scores = {}
    for strat_name, strat_data in results.items():
        strategy = GeneticStrategy(strat_data["best_strategy"], memory_length=1)
        scores = {}
        
        for test_noise in noise_levels:
            test_game = NoisyIPDGame(rounds=200, noise_level=test_noise)
            total_score = 0
            
            for opponent in opponents:
                player_score, _ = test_game.play_game(strategy, opponent)
                total_score += player_score
            
            avg_score = total_score / len(opponents)
            scores[test_noise] = avg_score
            
        robustness_scores[strat_name] = scores
    
    plt.figure(figsize=(10, 6))
    for strat_name, scores in robustness_scores.items():
        plt.plot(list(scores.keys()), list(scores.values()), marker='o', label=strat_name)
    plt.xlabel("Test Noise Level")
    plt.ylabel("Average Score")
    plt.title("Robustness of Evolved Strategies to Noise")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results, robustness_scores