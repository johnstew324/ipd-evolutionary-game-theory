import numpy as np
import matplotlib.pyplot as plt
import os
from IPD.strategies import AlwaysCooperate, AlwaysDefect, TitForTat, RandomStrategy, SuspiciousTitForTat, GrudgerStrategy,TitForTwoTats, GeneticStrategy
from IPD.game import IPDGame
from IPD.genetic_algorithm import IPDGeneticAlgorithm
from IPD.evolution import run_simple_evolution, run_extended_evolution, run_probabilistic_evolution, comparison_experiments
from IPD.noise_experiment import run_noise_experiment 

# Define results directory for saving outputs 
SAVE_PLOTS = True  
PLOT_DIR = "plots"  
if SAVE_PLOTS and not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def analyse_best_strategy(ga: IPDGeneticAlgorithm, rounds_per_match: int = 200) -> None:
    if ga.best_solution is None:
        print("No best solution found. Run evolve() first.")
        return
    best_strategy = ga.create_strategy(ga.best_solution)
    game = IPDGame(rounds=rounds_per_match)

    opponents = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        RandomStrategy(),
        SuspiciousTitForTat(),
        GrudgerStrategy(),
        TitForTwoTats(),
    ]
    results = {}
    print("\nDetailed Strategy Analysis:")
    print("-" * 50)
    print(f"Strategy: {ga.get_strategy_description()}")
    print("-" * 50)
    
    total_score = 0
    opponent_total = 0
    for opponent in opponents:
        best_strategy.reset()
        opponent.reset()
        
        my_history = []
        opp_history = []
        my_score = 0
        opp_score = 0
        
        for _ in range(rounds_per_match):
            my_move = best_strategy.make_move(opp_history)
            opp_move = opponent.make_move(my_history)
            
            my_history.append(my_move)
            opp_history.append(opp_move)
            
            round_score = game.play_round(my_move, opp_move)
            my_score += round_score[0]
            opp_score += round_score[1]
        
        cooperation_rate = sum(1 for move in my_history if move == 0) / len(my_history)
        mutual_cooperation = sum(1 for i in range(len(my_history)) if my_history[i] == 0 and opp_history[i] == 0)
        mutual_defection = sum(1 for i in range(len(my_history)) if my_history[i] == 1 and opp_history[i] == 1)
        exploited = sum(1 for i in range(len(my_history)) if my_history[i] == 0 and opp_history[i] == 1)
        exploiting = sum(1 for i in range(len(my_history)) if my_history[i] == 1 and opp_history[i] == 0)

        results[opponent.name] = {
            'my_score': my_score,
            'opp_score': opp_score,
            'cooperation_rate': cooperation_rate,
            'mutual_cooperation': mutual_cooperation,
            'mutual_defection': mutual_defection,
            'exploited': exploited,
            'exploiting': exploiting
        }
        total_score += my_score
        opponent_total += opp_score

        print(f"\nVs. {opponent.name}:")
        print(f"  Score: {my_score} vs {opp_score}")
        print(f"  Cooperation Rate: {cooperation_rate:.2f}")
        print(f"  Mutual Cooperation: {mutual_cooperation} rounds")
        print(f"  Mutual Defection: {mutual_defection} rounds")
        print(f"  Times Exploited (C,D): {exploited} rounds")
        print(f"  Times Exploiting (D,C): {exploiting} rounds")
    
    print("\nOverall Performance:")
    print(f"Total Score: {total_score} (vs {opponent_total} for opponents)")
    print(f"Average Score Per Opponent: {total_score / len(opponents):.2f}")

    plt.figure(figsize=(12, 6))
    opponent_names = [opp.name for opp in opponents]
    coop_rates = [results[opp.name]['cooperation_rate'] for opp in opponents]
    
    plt.bar(opponent_names, coop_rates)
    plt.xlabel('Opponent')
    plt.ylabel('Cooperation Rate')
    plt.title('Cooperation Rate Against Different Opponents')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if SAVE_PLOTS:
        filename = f"{PLOT_DIR}/cooperation_rate_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")      
    else:
        plt.show()
    plt.close()

    interaction_types = ['Mutual Cooperation', 'Mutual Defection', 'Exploited', 'Exploiting']
    plt.figure(figsize=(14, 8))
    
    x = np.arange(len(opponent_names))
    width = 0.2
    for i, interaction in enumerate(interaction_types):
        values = [results[opp.name][interaction.lower().replace(' ', '_')] for opp in opponents]
        plt.bar(x + i*width - width*1.5, values, width, label=interaction)
    
    plt.xlabel('Opponent')
    plt.ylabel('Number of Rounds')
    plt.title('Interaction Types Against Different Opponents')
    plt.xticks(x, opponent_names, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    if SAVE_PLOTS:
        filename = f"{PLOT_DIR}/interaction_type_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")      
    else:
        plt.show()
    plt.close()

    return results


def main():
    print("-" * 100)
    print("Iterated Prisoner's Dilemma - Evolutionary Algorithm")
    
# change all expect run_noise to false to run Part 2 of assignment (only)
    run_mem1 = True
    run_mem2 = True
    run_prob = True 
    run_comparison = True 
    run_noise = True #part 2 
    
    ga_list = []
    labels = []
    
    if run_mem1:
        print("\nRunning evolution with memory length 1")
        ga1 = run_simple_evolution()
        ga_list.append(ga1)
        labels.append("Memory Length 1")
        analyse_best_strategy(ga1)
    
    if run_mem2:
        print("\nRunning evolution with memory length 2")
        ga2 = run_extended_evolution()
        ga_list.append(ga2)
        labels.append("Memory Length 2")
    
    if run_prob:
        print("\nRunning evolution with probabilistic strategies")
        ga3 = run_probabilistic_evolution()
        ga_list.append(ga3)
        labels.append("Probabilistic Strategies")
    
    if run_comparison:
        print("\nRunning comparison experiments...")
        comparison_results = comparison_experiments()
        
    if run_noise:
        print("\n#### Noise Experiment ####")
        noise_results, robustness_scores = run_noise_experiment([0.0, 0.05, 0.1, 0.2])

    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()