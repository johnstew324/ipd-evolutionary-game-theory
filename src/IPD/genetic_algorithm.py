import numpy as np
import time
import os
import matplotlib.pyplot as plt
from .strategies import IPDStrategy, GeneticStrategy, ProbabilisticGeneticStrategy
from .game import IPDGame

SAVE_PLOTS = True  
PLOT_DIR = "plots" 

if SAVE_PLOTS and not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


class IPDGeneticAlgorithm:

    
    def __init__(self,
                 opponents,
                 population_size: int = 100,
                 elite_size: int = 10,
                 tournament_size: int = 5,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 rounds_per_game: int = 200,
                 memory_length: int = 1,
                 use_probabilities: bool = False):
        
        self.opponents = opponents
        self.population_size = population_size
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.memory_length = memory_length
        self.use_probabilities = use_probabilities
        
        self.game = IPDGame(rounds=rounds_per_game)
        
        if memory_length == 1:
            self.genome_size = 3
        elif memory_length == 2:
            self.genome_size = 5
        else:
            raise ValueError(f"Unsupported memory length: {memory_length}")

        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.best_solution = None
        self.best_fitness = 0  # 
    
    def initialise_population(self):
        if self.use_probabilities:
            return [np.random.random(self.genome_size) for _ in range(self.population_size)]
        else:
            return [np.random.randint(0, 2, self.genome_size) for _ in range(self.population_size)]
    
    def create_strategy(self, genome: np.ndarray) -> IPDStrategy:
        if self.use_probabilities:
            return ProbabilisticGeneticStrategy(genome, self.memory_length)
        else:
            return GeneticStrategy(genome, self.memory_length)
    
    def calculate_fitness(self, genome):
        strategy = self.create_strategy(genome)
        total_score = 0
        for opponent in self.opponents:
            player_score, _ = self.game.play_game(strategy, opponent)
            total_score += player_score
        
        return total_score
    
    def tournament_selection(self, population):
        tournament_idx = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament = [population[i] for i in tournament_idx]
        
        return max(tournament, key=self.calculate_fitness)
    
    def roulette_wheel_selection(self, population):
        fitness_values = [self.calculate_fitness(ind) for ind in population]
        total_fitness = sum(fitness_values)
        probabilities = [fit/total_fitness for fit in fitness_values]
        selected_idx = np.random.choice(len(population), p=probabilities)
        return population[selected_idx]
    
    def elitism_selection(self, population, num_elites): 
        fitness_values = [self.calculate_fitness(ind) for ind in population]
        sorted_indices = np.argsort(fitness_values)[::-1]
        elite_indices = sorted_indices[:num_elites]
        return [population[i] for i in elite_indices]
    
    def crossover(self, parent1, parent2):
        if np.random.random() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.genome_size)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            return child
        else:
            return parent1.copy()
    
    def mutate(self, genome):
        mutated_genome = genome.copy()
        
        for i in range(self.genome_size):
            if np.random.random() < self.mutation_rate:
                if self.use_probabilities:
                    mutated_genome[i] = np.clip(mutated_genome[i] + np.random.normal(0, 0.2), 0, 1)
                else:
                    mutated_genome[i] = 1 - mutated_genome[i]
        
        return mutated_genome
    
    def evolve(self, generations):
        start_time = time.time()
        population = self.initialise_population()
        
        for gen in range(generations):
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
                print(f"Generation {gen + 1}/{generations}, Best Fitness: {gen_best_fitness:.2f}, Avg Fitness: {gen_avg_fitness:.2f}")
        
        execution_time = time.time() - start_time

        return {
            'best_solution': self.best_solution,
            'best_fitness': self.best_fitness,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'execution_time': execution_time
        }
    
    def analyse_best_strategy(self):
        if self.best_solution is None:
            raise ValueError("No best solution found. Run evolve() first.")
        best_strategy = self.create_strategy(self.best_solution)
        results = {}
        for opponent in self.opponents:
            player_score, opponent_score = self.game.play_game(best_strategy, opponent)
            results[opponent.name] = {
                'player_score': player_score,
                'opponent_score': opponent_score
            }
        return results
    
    def plot_fitness_history(self, title: str = "Fitness History"):
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='Best Fitness')
        plt.plot(self.avg_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Total Score)')
        plt.title(title)
        plt.legend()
        plt.grid(True)

        if SAVE_PLOTS:
            filename = f"{PLOT_DIR}/{title.replace(' ', '_').lower()}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved plot to {filename}")
        else:
            plt.show()
        plt.close()

        



    
    def get_strategy_description(self):
        if self.best_solution is None:
            return "No strategy evolved yet"
        if self.memory_length == 1:
            if self.use_probabilities:
                return (f"First move: {self.best_solution[0]:.2f} probability to cooperate\n"
                        f"If opponent cooperated: {self.best_solution[1]:.2f} probability to cooperate\n"
                        f"If opponent defected: {self.best_solution[2]:.2f} probability to cooperate")
            else:
                first_move = "Cooperate" if self.best_solution[0] == 0 else "Defect"
                after_c = "Cooperate" if self.best_solution[1] == 0 else "Defect"
                after_d = "Cooperate" if self.best_solution[2] == 0 else "Defect"
                return (f"First move: {first_move}\n"
                        f"If opponent cooperated: {after_c}\n"
                        f"If opponent defected: {after_d}")
    
        elif self.memory_length == 2:
            if self.use_probabilities:
                return (f"First move: {self.best_solution[0]:.2f} probability to cooperate\n"
                        f"If opponent played CC: {self.best_solution[1]:.2f} probability to cooperate\n"
                        f"If opponent played CD: {self.best_solution[2]:.2f} probability to cooperate\n"
                        f"If opponent played DC: {self.best_solution[3]:.2f} probability to cooperate\n"
                        f"If opponent played DD: {self.best_solution[4]:.2f} probability to cooperate")
            else:
                first_move = "Cooperate" if self.best_solution[0] == 0 else "Defect"
                after_cc = "Cooperate" if self.best_solution[1] == 0 else "Defect"
                after_cd = "Cooperate" if self.best_solution[2] == 0 else "Defect"
                after_dc = "Cooperate" if self.best_solution[3] == 0 else "Defect"
                after_dd = "Cooperate" if self.best_solution[4] == 0 else "Defect"
                
                return (f"First move: {first_move}\n"
                        f"If opponent played CC: {after_cc}\n"
                        f"If opponent played CD: {after_cd}\n"
                        f"If opponent played DC: {after_dc}\n"
                        f"If opponent played DD: {after_dd}")