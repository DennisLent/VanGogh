import numpy as np

from vangogh.population import Population


def select(population, selection_size, selection_name='tournament_4'):
    if 'tournament' in selection_name:
        tournament_size = int(selection_name.split('_')[-1])
        return tournament_select(population, selection_size, tournament_size)
    elif selection_name == 'roulette_wheel':
        return roulette_wheel_select(population, selection_size)
    elif selection_name == 'rank':
        return rank_select(population, selection_size)
    else:
        raise ValueError('Invalid selection name:', selection_name)


def one_tournament_round(population, tournament_size, return_winner_index=False):
    rand_perm = np.random.permutation(len(population.fitnesses))
    competing_fitnesses = population.fitnesses[rand_perm[:tournament_size]]
    winning_index = rand_perm[np.argmin(competing_fitnesses)]
    if return_winner_index:
        return winning_index
    else:
        return {
            'genotype': population.genes[winning_index, :],
            'fitness': population.fitnesses[winning_index],
        }


def tournament_select(population, selection_size, tournament_size=4):
    genotype_length = population.genes.shape[1]
    selected = Population(selection_size, genotype_length, "N/A")

    n = len(population.fitnesses)
    num_selected_per_iteration = n // tournament_size
    num_parses = selection_size // num_selected_per_iteration

    for i in range(num_parses):
        # shuffle
        population.shuffle()

        winning_indices = np.argmin(population.fitnesses.squeeze().reshape((-1, tournament_size)),
                                    axis=1)
        winning_indices += np.arange(0, n, tournament_size)

        selected.genes[i * num_selected_per_iteration:(i + 1) * num_selected_per_iteration,
        :] = population.genes[winning_indices, :]
        selected.fitnesses[i * num_selected_per_iteration:(i + 1) * num_selected_per_iteration] = \
        population.fitnesses[winning_indices]

    return selected

def roulette_wheel_select(population, selection_size):
    selected = Population(selection_size, population.genes.shape[1], "N/A")

    # Adjust fitness values for minimization
    adjusted_fitness = np.max(population.fitnesses) - population.fitnesses + 1

    # Normalize the fitness values to make them proportional to probabilities
    probabilities = adjusted_fitness / np.sum(adjusted_fitness)

    # Perform selection
    selected_indices = np.random.choice(np.arange(len(population.genes)), size=selection_size, p=probabilities)
    selected.genes = population.genes[selected_indices, :]
    selected.fitnesses = population.fitnesses[selected_indices]

    return selected

def rank_select(population, selection_size):
    selected = Population(selection_size, population.genes.shape[1], "N/A")

    # Rank the population in reverse (smaller fitness has higher rank)
    ranked_indices = np.argsort(population.fitnesses)[::-1]

    # Compute the rank-based probabilities
    probabilities = np.arange(len(population.fitnesses), 0, -1)
    probabilities = probabilities / np.sum(probabilities)

    # Perform selection
    selected_indices = np.random.choice(ranked_indices, size=selection_size, p=probabilities)
    selected.genes = population.genes[selected_indices, :]
    selected.fitnesses = population.fitnesses[selected_indices]

    return selected

