import numpy as np


def crossover(genes, method="ONE_POINT"):
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
    else:
        raise Exception("Unknown crossover method")

    return offspring


def mutate(genes, feature_intervals,
           mutation_probability=0.1, num_features_mutation_strength=0.05):
    
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
        range_num = feature_intervals[i][1] - feature_intervals[i][0]
        low = -num_features_mutation_strength / 2
        high = +num_features_mutation_strength / 2

        mutations[:, i] = range_num * np.random.uniform(low=low, high=high,
                                                        size=mutations.shape[0])
        mutations[:, i] += genes[:, i]

        # Fix out-of-range
        mutations[:, i] = np.where(mutations[:, i] > feature_intervals[i][1],
                                   feature_intervals[i][1], mutations[:, i])
        mutations[:, i] = np.where(mutations[:, i] < feature_intervals[i][0],
                                   feature_intervals[i][0], mutations[:, i])

    mutations = mutations.astype(int)
    return mutations
