import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image

# Load the image and convert to grayscale
image = Image.open("julia-ly-scJ9atGup7E-unsplash.jpg").convert("L")
resized_image = image.resize((128, 64))
image_array = np.array(resized_image)

# Function to split the image into patches of 2 columns and 4 rows
def split_into_patches(image, num_rows, num_cols):
    patches = []
    row_height = image.shape[0] // num_rows
    col_width = image.shape[1] // num_cols
    for i in range(0, image.shape[0], row_height):
        for j in range(0, image.shape[1], col_width):
            patch = image[i:i+row_height, j:j+col_width]
            patches.append(patch)
    return patches

# Create initial population
def create_initial_population(image, num_rows, num_cols, population_size):
    patches = split_into_patches(image, num_rows, num_cols)
    return [np.random.permutation(patches) for _ in range(population_size)]

# Fitness function (normalized to prevent overflow)
def calculate_fitness(individual, original_patches):
    fitness = 0
    for i in range(len(individual)):
        fitness -= np.sum(np.abs(individual[i] - original_patches[i])) / (individual[i].size)
    return fitness

# Apply elitism
def elitism(population, fitnesses, elite_size):
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
    return sorted_population[:elite_size]

# Apply Simulated Annealing
def simulated_annealing(individual, original_patches, initial_temp, cooling_rate):
    current_solution = individual.copy()
    current_fitness = calculate_fitness(current_solution, original_patches)
    temp = initial_temp

    while temp > 1:
        i, j = random.sample(range(len(current_solution)), 2)
        new_solution = current_solution.copy()
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]

        new_fitness = calculate_fitness(new_solution, original_patches)
        delta = new_fitness - current_fitness
        if delta > 0 or random.random() < np.exp(min(delta / temp, 700)):  # Limit to prevent overflow
            current_solution = new_solution
            current_fitness = new_fitness

        temp *= cooling_rate

    return current_solution

# Genetic algorithm
def genetic_algorithm(image_array, num_rows, num_cols, population_size, generations, elite_size, initial_mutation_rate, initial_temp, cooling_rate):
    population = create_initial_population(image_array, num_rows, num_cols, population_size)
    original_patches = split_into_patches(image_array, num_rows, num_cols)
    best_overall_individual = None
    best_overall_fitness = float('-inf')
    mutation_rate = initial_mutation_rate

    for generation in range(generations):
        fitnesses = [calculate_fitness(individual, original_patches) for individual in population]

        # Update the best individual
        if max(fitnesses) > best_overall_fitness:
            best_overall_fitness = max(fitnesses)
            best_overall_individual = population[np.argmax(fitnesses)]

        print(f"Generation {generation}: Best fitness = {best_overall_fitness}")

        # Save the best solution in each generation
        visualize_and_save_solution(best_overall_individual, num_rows, num_cols, image_array.shape, generation)

        # Apply elitism
        new_population = elitism(population, fitnesses, elite_size)

        # Create new population
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(new_population, 2)
            crossover_point = random.randint(1, len(parent1) - 1)
            child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

            if random.random() < mutation_rate:
                i, j = random.sample(range(len(child)), 2)
                child[i], child[j] = child[j], child[i]

            new_population.append(child)

        # Improve elite individuals with Simulated Annealing
        for i in range(elite_size):
            new_population[i] = simulated_annealing(new_population[i], original_patches, initial_temp, cooling_rate)

        population = new_population

        # Gradually change the mutation rate
        mutation_rate = min(1.0, mutation_rate * 1.1)

    return best_overall_individual

# Visualize and save the solution
def visualize_and_save_solution(solution, num_rows, num_cols, original_shape, generation):
    row_height, col_width = original_shape[0] // num_rows, original_shape[1] // num_cols
    reconstructed = np.zeros(original_shape, dtype=np.uint8)
    index = 0
    for i in range(0, original_shape[0], row_height):
        for j in range(0, original_shape[1], col_width):
            reconstructed[i:i+row_height, j:j+col_width] = solution[index]
            index += 1
    plt.imshow(reconstructed, cmap='gray')
    plt.axis('off')
    plt.title(f"Generation {generation}")
    plt.savefig(f"generation_{generation}.png")  # Save the image
    plt.close()

# Run the algorithm
best_solution = genetic_algorithm(image_array, 4, 4, 2000, 5, 20, 0.3, 1500, 0.9)

# Original image size
original_shape = image_array.shape
visualize_and_save_solution(best_solution, 4, 4, original_shape, "final")
