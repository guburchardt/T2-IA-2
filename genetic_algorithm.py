import numpy as np
from fitness_function import fitness_function

def genetic_algorithm(pop_size, weight_shape, generations, game_state, minimax_level):
    population = initialize_population(pop_size, weight_shape)

    for generation in range(generations):
        fitness_scores = [fitness_function(ind, game_state, minimax_level) for ind in population]

        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        print(f"Geração {generation + 1}/{generations}: Melhor Fitness = {best_fitness:.2f}, Fitness Médio = {avg_fitness:.2f}")

        new_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        elite = population[np.argmax(fitness_scores)]
        new_population[-1] = elite
        population = new_population

    best_idx = np.argmax(fitness_scores)
    return population[best_idx]

# Inicializar população
def initialize_population(pop_size, weight_shape):
    return [np.random.uniform(-1, 1, weight_shape) for _ in range(pop_size)]

# Seleção por torneio
def tournament_selection(population, fitness_scores, k=3):
    selected = np.random.choice(len(population), k, replace=False)
    best = max(selected, key=lambda i: fitness_scores[i])
    return population[best]

# Função de crossover
def crossover(parent1, parent2):
    alpha = np.random.uniform(0, 1)
    child1 = alpha * parent1 + (1 - alpha) * parent2
    child2 = (1 - alpha) * parent1 + alpha * parent2
    return child1, child2

# Função de mutação
def mutate(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            chromosome[i] += np.random.normal(0, 0.5)
    return chromosome
