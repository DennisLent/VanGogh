import numpy as np
from scipy.stats import qmc
from sklearn.cluster import KMeans

class Population:
    def __init__(self, population_size, genotype_length, initialization, image = None):
        self.genes = np.empty(shape=(population_size, genotype_length), dtype=int)
        self.fitnesses = np.zeros(shape=(population_size,))
        self.initialization = initialization
        self.image = image

    def initialize(self, feature_intervals, image_data):
        n = self.genes.shape[0]
        l = self.genes.shape[1]
        if self.initialization == "RANDOM":
            self.uniform_random_initialization(feature_intervals, n, l)
        elif self.initialization == "LHS":
            self.lhs_initialization(feature_intervals, n, l)
        elif self.initialization == "QUASIRANDOM":
            self.quasirandom_initialization(feature_intervals, n, l)
        elif self.initialization == "CLUSTER":
            self.cluster_initialization(n, l, feature_intervals)
        elif self.initialization == "MATCH":
            for i in range(l):
                init_feat_i = np.random.randint(low=feature_intervals[i][0],
                                                        high=feature_intervals[i][1], size=n)
                self.genes[:, i] = init_feat_i
            for g in range(0, len(self.genes)):
                for i in range(0, len(self.genes[g]), 5):
                    x, y = self.genes[g][i:i+2]
                    self.genes[g][i+2:i+5] = image_data[y][x]
        else:
            raise Exception("Unknown initialization method")

    def uniform_random_initialization(self, feature_intervals, n, l):
        for i in range(l):
            init_feat_i = np.random.randint(low=feature_intervals[i][0],
                                            high=feature_intervals[i][1], size=n)
            self.genes[:, i] = init_feat_i

    def lhs_initialization(self, feature_intervals, n, l):
        # Scipy's LHS doesn't support different intervals for each dimension
        # We'll generate for [0, 1] and then scale and shift
        lhs_sample = qmc.LatinHypercube(l).random(n)
        for i in range(l):
            a, b = feature_intervals[i]
            self.genes[:, i] = np.floor(a + lhs_sample[:, i] * (b - a)).astype(int)

    def quasirandom_initialization(self, feature_intervals, n, l):
        halton_sample = qmc.Halton(l).random(n)
        for i in range(l):
            a, b = feature_intervals[i]
            self.genes[:, i] = np.floor(a + halton_sample[:, i] * (b - a)).astype(int)

    def cluster_initialization(self, n, l, feature_intervals):
        if self.image is None:
            raise Exception("Image not provided for CLUSTER initialization")

        # Flatten image into a 2D array where each row is a color
        image_array = np.array(self.image).reshape(-1, 3)

        # Perform k-means clustering to find the most dominant colors
        kmeans = KMeans(n_clusters=n)  # number of clusters = number of individuals
        kmeans.fit(image_array)

        # Initialize genes based on k-means clustering results
        for i in range(l):
            lower, upper = feature_intervals[i]
            if i % 3 == 0:  # X
                init_feat_i = np.random.randint(low=lower, high=upper, size=n)
            elif i % 3 == 1:  # Y
                init_feat_i = np.random.randint(low=lower, high=upper, size=n)
            else:  # color (RGB)
                init_feat_i = kmeans.cluster_centers_[:, i % 3]
                init_feat_i = np.clip(init_feat_i, lower, upper).astype(int)
            self.genes[:, i] = init_feat_i

    def stack(self, other):
        self.genes = np.vstack((self.genes, other.genes))
        self.fitnesses = np.concatenate((self.fitnesses, other.fitnesses))

    def shuffle(self):
        random_order = np.random.permutation(self.genes.shape[0])
        self.genes = self.genes[random_order, :]
        self.fitnesses = self.fitnesses[random_order]

    def is_converged(self):
        return len(np.unique(self.genes, axis=0)) < 2

    def delete(self, indices):
        self.genes = np.delete(self.genes, indices, axis=0)
        self.fitnesses = np.delete(self.fitnesses, indices)
