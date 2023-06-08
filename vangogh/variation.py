import numpy as np
from vangogh.util import NUM_VARIABLES_PER_POINT


def crossover(genes, fitness_scores, method="ONE_POINT"):
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))

    offspring = np.zeros(shape=genes.shape, dtype=int)

    if method == "ONE_POINT":
        crossover_points = np.random.randint(0, genes.shape[1], size=genes.shape[0])
        for i in range(len(genes)):
            offspring[i,:] = np.where(np.arange(genes.shape[1]) <= crossover_points[i], parents_1[i,:], parents_2[i,:])
    elif method == "TWO_POINT":
        crossover_points = np.sort(np.random.randint(0, genes.shape[1], size=(genes.shape[0], 2)), axis=1)
        for i in range(len(genes)):
            mask = np.logical_and(np.arange(genes.shape[1]) >= crossover_points[i, 0], np.arange(genes.shape[1]) <= crossover_points[i, 1])
            offspring[i,:] = np.where(mask, parents_1[i,:], parents_2[i,:])
    elif method == "THREE_POINT":
        crossover_points = np.sort(np.random.randint(0, genes.shape[1], size=(genes.shape[0], 3)), axis=1)
        for i in range(len(genes)):
            mask1 = np.logical_and(np.arange(genes.shape[1]) >= crossover_points[i, 0], np.arange(genes.shape[1]) < crossover_points[i, 1])
            mask2 = np.logical_and(np.arange(genes.shape[1]) >= crossover_points[i, 1], np.arange(genes.shape[1]) <= crossover_points[i, 2])
            offspring[i,:] = np.where(mask1, parents_1[i,:], np.where(mask2, parents_2[i,:], parents_1[i,:]))
    elif method == "UNIFORM":
        for i in range(len(genes)):
            mask = np.random.choice([True, False], size=genes.shape[1])
            offspring[i,:] = np.where(mask, parents_1[i,:], parents_2[i,:])
    elif method == "OPTIMAL_MIXING":
        offspring = optimal_mixing(genes, fitness_scores)
    else:
        raise Exception("Unknown crossover method")

    return offspring

def optimal_mixing(genes, fitness_scores):
    # Start with the first parent's genes
    offspring = genes[:len(genes)//2].copy()

    # For each gene
    for i in range(genes.shape[1]):
        # Select the genes from the parents
        gene_parent_1 = genes[:len(genes)//2, i]
        gene_parent_2 = genes[len(genes)//2:, i]

        # Select the fitness of the parents
        fitness_parent_1 = fitness_scores[:len(genes)//2]
        fitness_parent_2 = fitness_scores[len(genes)//2:]

        # Calculate the probabilities for each parent
        probabilities = fitness_parent_1 / (fitness_parent_1 + fitness_parent_2)

        # Choose which parent's gene to keep
        mask = np.random.uniform(size=len(gene_parent_1)) < probabilities

        # Apply the mask to the offspring genes
        offspring[:, i] = np.where(mask, gene_parent_1, gene_parent_2)

    return offspring


def mutate(genes, feature_intervals,
           mutation_probability=0.1, num_features_mutation_strength=0.05, method = "ADD_RANDOM"):
    
    
    if method == "ADD_RANDOM":
        return add_random_mutation(genes, feature_intervals, mutation_probability,
                                   num_features_mutation_strength)
    elif method == "PERMUTATION":
        return permutation_mutation(genes, mutation_probability)
    elif method == "SHRINK":
        return shrink_mutation(genes, feature_intervals, mutation_probability)
    elif method == "GAUSSIAN":
        return gaussian_mutation(genes, feature_intervals, mutation_probability)
    elif method == "SCRAMBLE":
        return scramble_mutation(genes, mutation_probability)
    else:
        raise Exception("Unknown mutation method")

def shrink_mutation(genes, feature_intervals, mutation_probability):
    shrink_factor = 0.9
    mask_mut = np.random.choice([True, False], size=genes.shape,
                                    p=[mutation_probability, 1 - mutation_probability])
        
        # Shrink mutation
    mutations = genes * shrink_factor
        
        # Ensure mutations are within the allowable range and cast to integers
    mutations = np.clip(mutations, feature_intervals[:, 0], feature_intervals[:, 1])
    mutations = mutations.astype(int)

        # Apply the mutations where the mask is True
    offspring = np.where(mask_mut, mutations, genes)
    return offspring
    

def add_random_mutation(genes, feature_intervals, mutation_probability, num_features_mutation_strength):
    mask_mut = np.random.choice([True, False], size=genes.shape,
                                p=[mutation_probability, 1 - mutation_probability])

    mutations = generate_plausible_mutations(genes, feature_intervals,
                                             num_features_mutation_strength)

    offspring = np.where(mask_mut, mutations, genes)

    return offspring

def generate_plausible_mutations(genes, feature_intervals,
                                 num_features_mutation_strength=0.25):
    mutations = np.zeros(shape=genes.shape)

    for i in range(genes.shape[1]):
        range_num = feature_intervals[i % NUM_VARIABLES_PER_POINT][1] - feature_intervals[i % NUM_VARIABLES_PER_POINT][0]
        low = -num_features_mutation_strength / 2
        high = +num_features_mutation_strength / 2

        mutations[:, i] = range_num * np.random.uniform(low=low, high=high,
                                                        size=mutations.shape[0])
        mutations[:, i] += genes[:, i]

        # Fix out-of-range
        mutations[:, i] = np.where(mutations[:, i] > feature_intervals[i % NUM_VARIABLES_PER_POINT][1],
                                   feature_intervals[i % NUM_VARIABLES_PER_POINT][1], mutations[:, i])
        mutations[:, i] = np.where(mutations[:, i] < feature_intervals[i % NUM_VARIABLES_PER_POINT][0],
                                   feature_intervals[i % NUM_VARIABLES_PER_POINT][0], mutations[:, i])

    mutations = mutations.astype(int)
    return mutations

def gaussian_mutation(genes, feature_intervals, mutation_probability):
    genes = np.array(genes, dtype=int)
    mask_mut = np.random.choice([True, False], size=genes.shape,
                                p=[mutation_probability, 1 - mutation_probability])

    for i in range(genes.shape[1]):
        range_num = feature_intervals[i % NUM_VARIABLES_PER_POINT][1] - feature_intervals[i % NUM_VARIABLES_PER_POINT][0]
        std_dev = range_num / 6
        mutations[:, i] = np.random.normal(loc=genes[:, i], scale=std_dev)

    mutations = np.clip(mutations, feature_intervals[:,0], feature_intervals[:,1])

    offspring = np.where(mask_mut, mutations.astype(int), genes)

    return offspring

def scramble_mutation(genes, mutation_probability):
    mask_mut = np.random.choice([True, False], size=genes.shape[0],
                                p=[mutation_probability, 1 - mutation_probability])

    offspring = genes.copy()

    for i in range(genes.shape[0]):
        if mask_mut[i]:
            scramble_range = np.sort(np.random.randint(0, genes.shape[1], 2))
            scrambled = np.random.permutation(genes[i, scramble_range[0]:scramble_range[1]])
            offspring[i, scramble_range[0]:scramble_range[1]] = scrambled

    return offspring


def permutation_mutation(genes, mutation_probability=0.1):
    offspring = genes.copy()

    for i in range(len(genes)):
        if np.random.rand() < mutation_probability:
            # Select two random indices in the gene and swap them
            idx1, idx2 = np.random.choice(genes.shape[1], 2, replace=False)
            offspring[i, idx1], offspring[i, idx2] = offspring[i, idx2], offspring[i, idx1]

    return offspring


