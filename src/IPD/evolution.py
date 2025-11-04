import numpy as np
import matplotlib.pyplot as plt
import os  
from .strategies import AlwaysCooperate, AlwaysDefect, TitForTat, RandomStrategy, SuspiciousTitForTat, GrudgerStrategy,TitForTwoTats, GeneticStrategy
from .game import IPDGame
from .genetic_algorithm import IPDGeneticAlgorithm

# Define results directory for saving outputs
SAVE_PLOTS = True  # set False to disable saving
PLOT_DIR = "plots"  # all images go here
if SAVE_PLOTS and not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

SAVE_PLOTS = True  # set False to disable saving
PLOT_DIR = "plots"  # all images go here
if SAVE_PLOTS and not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)

def run_simple_evolution():
    opponents = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        RandomStrategy(),
        SuspiciousTitForTat(),
        GrudgerStrategy(),
        TitForTwoTats()
    ]

    ga = IPDGeneticAlgorithm(
        opponents=opponents,
        population_size=100,
        elite_size=10,
        tournament_size=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        memory_length=1,
        use_probabilities=False  
    )
    

    print("Running evolution with memory length 1...")
    results = ga.evolve(generations=100)

    print("\nBest Strategy:")
    print(ga.get_strategy_description())
    
    print("\nPerformance against opponents:")
    opponent_results = ga.analyse_best_strategy()
    for opponent, scores in opponent_results.items():
        print(f"{opponent}: {scores['player_score']} vs {scores['opponent_score']}")
    ga.plot_fitness_history("Evolution with Memory Length 1")
    
    return ga

def run_extended_evolution():
    opponents = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        RandomStrategy(),
        SuspiciousTitForTat(),
        GrudgerStrategy(),
        TitForTwoTats(),
    ]
    
    ga = IPDGeneticAlgorithm(
        opponents=opponents,
        population_size=100,
        elite_size=10,
        tournament_size=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        memory_length=2,
        use_probabilities=False  
    )
    
    print("Running evolution with memory length 2...")
    results = ga.evolve(generations=100)
    
    print("\nBest Strategy (Memory Length 2):")
    print(ga.get_strategy_description())
    
    print("\nPerformance against opponents:")
    opponent_results = ga.analyse_best_strategy()
    for opponent, scores in opponent_results.items():
        print(f"{opponent}: {scores['player_score']} vs {scores['opponent_score']}")
    
    ga.plot_fitness_history("Evolution with Memory Length 2")
    
    return ga

def run_probabilistic_evolution():
    opponents = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        RandomStrategy(),
        SuspiciousTitForTat(),
        GrudgerStrategy(),
        TitForTwoTats(),
    ]
    
    ga = IPDGeneticAlgorithm(
        opponents=opponents,
        population_size=100,
        elite_size=10,
        tournament_size=5,
        mutation_rate=0.1,
        crossover_rate=0.7,
        memory_length=1,
        use_probabilities=True 
    )
    
    print("Running evolution with probabilistic strategies...")
    results = ga.evolve(generations=100)
    print("\nBest Probabilistic Strategy:")
    print(ga.get_strategy_description())
    
    print("\nPerformance against opponents:")
    opponent_results = ga.analyse_best_strategy()
    for opponent, scores in opponent_results.items():
        print(f"{opponent}: {scores['player_score']} vs {scores['opponent_score']}")
    
    ga.plot_fitness_history("Evolution with Probabilistic Strategies")
    return ga

def comparison_experiments():
    experiment_configs = [
        {
            "name": "All Cooperative",
            "opponents": [AlwaysCooperate()],
        },
        {
            "name": "All Defective",
            "opponents": [AlwaysDefect()],
        },
        {
            "name": "Tit for Tat",
            "opponents": [TitForTat()],
        },
        {
            "name": "Mixed Simple",
            "opponents": [AlwaysCooperate(), AlwaysDefect(), TitForTat()],
        },
        {
            "name": "Mixed Complex",
            "opponents": [
                TitForTat(), 
                SuspiciousTitForTat(), 
                GrudgerStrategy(), 
                TitForTwoTats(), 
            ],
        }
    ]
    results = {}
    
    for config in experiment_configs:
        print(f"\nRunning experiment: {config['name']}")
        
        ga = IPDGeneticAlgorithm(
            opponents=config["opponents"],
            population_size=100,
            elite_size=10,
            tournament_size=5,
            mutation_rate=0.1,
            crossover_rate=0.7,
            memory_length=1,
            use_probabilities=False
        )
        

        ga_results = ga.evolve(generations=50)
        
        results[config["name"]] = {
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
    plt.title("Comparison of Evolution Against Different Opponents")
    plt.legend()
    plt.grid(True)
    if SAVE_PLOTS:
        filename = f"{PLOT_DIR}/comparison_of_evolution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show()
    plt.close()

    
    return results

def run_diversity_analysis():
    opponents = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        RandomStrategy()
    ]
    
    class DiversityTrackingGA(IPDGeneticAlgorithm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.diversity_history = []
        
        def measure_diversity(self, population):
            if self.use_probabilities:
                genome_array = np.array(population)
                return np.mean(np.std(genome_array, axis=0))
            else:
                unique_strategies = set(tuple(genome) for genome in population)
                return len(unique_strategies) / self.population_size
        
        def evolve(self, generations=100):
            population = self.initialise_population()
            
            for gen in range(generations):
                diversity = self.measure_diversity(population)
                self.diversity_history.append(diversity)
                
                fitness_scores = [self.calculate_fitness(ind) for ind in population]
                gen_best_fitness = max(fitness_scores)
                gen_avg_fitness = sum(fitness_scores) / len(fitness_scores)
                
                self.best_fitness_history.append(gen_best_fitness)
                self.avg_fitness_history.append(gen_avg_fitness)
                
                if gen_best_fitness > self.best_fitness:
                    self.best_fitness = gen_best_fitness
                    self.best_solution = population[fitness_scores.index(gen_best_fitness)]
                
                elites = self.elitism_selection(population, self.elite_size)
                new_population = elites.copy()
                
                while len(new_population) < self.population_size:
                    parent1 = self.tournament_selection(population)
                    parent2 = self.tournament_selection(population)
                    child = self.crossover(parent1, parent2)
                    child = self.mutate(child)
                    new_population.append(child)
                
                population = new_population
                
                if (gen + 1) % 10 == 0:
                    print(f"Generation {gen + 1}/{generations}, Diversity: {diversity:.4f}")
            
            return {
                'best_solution': self.best_solution,
                'best_fitness': self.best_fitness,
                'diversity_history': self.diversity_history
            }
    mutation_rates = [0.01, 0.05, 0.1, 0.2]
    diversity_results = {}
    
    for rate in mutation_rates:
        print(f"\nRunning with mutation rate: {rate}")
        
        ga = DiversityTrackingGA(
            opponents=opponents,
            population_size=100,
            elite_size=10,
            tournament_size=5,
            mutation_rate=rate,
            crossover_rate=0.7,
            memory_length=1,
            use_probabilities=False
        )
        
        results = ga.evolve(generations=50)
        diversity_results[f"Mutation Rate {rate}"] = {
            "diversity": results["diversity_history"],
            "fitness": ga.best_fitness_history
        }
    
    plt.figure(figsize=(12, 6))
    for label, data in diversity_results.items():
        plt.plot(data["diversity"], label=label)
    
    plt.xlabel("Generation")
    plt.ylabel("Population Diversity")
    plt.title("Population Diversity During Evolution")
    plt.legend()
    plt.grid(True)
    if SAVE_PLOTS:
        filename = f"{PLOT_DIR}/population_diversity_during_evolution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show()
    plt.close()
    
    return diversity_results

def run_opponent_response_analysis(ga: IPDGeneticAlgorithm, rounds: int = 200):
    if ga.best_solution is None:
        print("No best solution found. Run evolve() first.")
        return None
    
    best_strategy = ga.create_strategy(ga.best_solution)
    
    opponents = [
        AlwaysCooperate(),
        AlwaysDefect(),
        TitForTat(),
        RandomStrategy(),
        SuspiciousTitForTat(),
        GrudgerStrategy()
    ]
    
    results = {}
    
    for opponent in opponents:
        best_strategy.reset()
        opponent.reset()
        
        my_history = []
        opp_history = []
        
        for _ in range(rounds):
            my_move = best_strategy.make_move(opp_history)
            opp_move = opponent.make_move(my_history)
            
            my_history.append(my_move)
            opp_history.append(opp_move)

        after_c_indices = [i+1 for i in range(len(opp_history)-1) if opp_history[i] == 0]
        after_c_responses = [my_history[i] for i in after_c_indices if i < len(my_history)]
        c_after_c = sum(1 for m in after_c_responses if m == 0) / max(1, len(after_c_responses))

        after_d_indices = [i+1 for i in range(len(opp_history)-1) if opp_history[i] == 1]
        after_d_responses = [my_history[i] for i in after_d_indices if i < len(my_history)]
        c_after_d = sum(1 for m in after_d_responses if m == 0) / max(1, len(after_d_responses))

        results[opponent.name] = {
            'c_after_c': c_after_c,
            'c_after_d': c_after_d,
            'my_history': my_history,
            'opp_history': opp_history
        }

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    opponent_names = list(results.keys())
    c_after_c_values = [results[name]['c_after_c'] for name in opponent_names]
    
    plt.bar(opponent_names, c_after_c_values)
    plt.title('Cooperation Rate After Opponent Cooperation')
    plt.ylabel('Cooperation Probability')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    plt.subplot(1, 2, 2)
    c_after_d_values = [results[name]['c_after_d'] for name in opponent_names]
    
    plt.bar(opponent_names, c_after_d_values)
    plt.title('Cooperation Rate After Opponent Defection')
    plt.ylabel('Cooperation Probability')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    if SAVE_PLOTS:
        filename = f"{PLOT_DIR}/opponent_response_analysis.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {filename}")
    else:
        plt.show()
    plt.close()

    return results
