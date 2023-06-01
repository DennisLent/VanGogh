# Evolutionary Van Gogh

To run this code, install the required packages from requirements.txt and open the Jupyter notebook (`analysis.ipynb`) for an example of how to run the code and experiments with it.

```bash
pip install -r requirements.txt
```

To start a Jupyter notebook instance, have Jupyter notebook [installed](https://jupyter.org/install#jupyter-notebook) and start it up in this directory.

## Improvements to the code
1. Population initialization: Adaptive initialization methods adjust the population initialization strategy based on prior knowledge or information gathered during the optimization process. 
2. Tournament Selection: Adaptive Tournament size: instead of using a fixed tournament size, adapting the tournament size during process.  
3. Hybrid Selection Schemes: Consider combining tournament selection with other selection methods (using a combination of tournament selection and elitism)
4. Variation: (Exploration-Exploitation Balance) Adjust the mutation probability and mutation strength parameters 

## Feedback from Damy
- Besides the provided "ONE_POINT" crossover, we can explore other crossover methods such as "UNIFORM" or "TWO_POINT" to introduce additional variation. However we should consider using more novel ideas (like three point cross-over)
- Fitness-based Variation: Modify the mutation and crossover operators to take into account the fitness information. E.g. design mutation operators that prioritize changes in less fit individuals
- Population initialization → deduce some statistics from the reference image, uniform initialization	
- We should look into model-based EAs

## Notes 01.06
- implemented uniform, 2 and 3 point crossover -> uniform is the best so far (42k down to 34k) 
- increasing population size is beneficial (100 to 300) but increases compution time
- tried to consider fitness based on mutation (higher in the beginning and lower later)
- could potentially implement elitism for variation
- could try new initialization method
- could work with adaptive mutations
- could check into variations.py/generate_plausible_mutations

### Hypothese
- we think that uniform cross-over will improve the fitness
- we think elitism will improve the convergence rate, but not necessarily improve the fitness
- Changing the mutation probability, such that in the beginning its higher, will produce better fitness results.
