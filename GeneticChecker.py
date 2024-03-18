import random
import networkx as nx
import matplotlib.pyplot as plt
import time

def read_edge_list(file_path):
    with open(file_path, 'r') as file:
        edges = [tuple(map(int, line.strip().split())) for line in file]
    return edges

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        mutation_points = random.sample(range(len(individual)), 2)
        individual[mutation_points[0]], individual[mutation_points[1]] = individual[mutation_points[1]], individual[mutation_points[0]]
    return individual

def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
    return child

def is_hamiltonian_cycle(graph, individual):
    return len(set(individual)) == len(graph) == len(individual)

def evolve(graph, population_size, generations, mutation_rate):
    start_time = time.time()
    population = [random.sample(graph.keys(), len(graph)) for _ in range(population_size)]

    for generation in range(generations):
        population = sorted(population, key=lambda ind: is_hamiltonian_cycle(graph, ind))
        if is_hamiltonian_cycle(graph, population[0]):
            elapsed_time = time.time() - start_time
            return population[0], elapsed_time  # Hamiltonian cycle found

        parents = population[:population_size // 2]

        offspring = []
        while len(offspring) < population_size - len(parents):
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            offspring.append(child)

        population = parents + offspring

    elapsed_time = time.time() - start_time
    return None, elapsed_time  # No Hamiltonian cycle found within the specified generations

def draw_graph(graph, title="Graph"):
    G = nx.Graph()
    for node, neighbors in graph.items():
        G.add_edges_from([(node, neighbor) for neighbor in neighbors])

    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.title(title)
    plt.show()

# Example usage:
file_path = "II_7932.hcp.txt"  # Replace with the path to your edge list file
edges = read_edge_list(file_path)
graph_from_edge_list = nx.Graph(edges)
graph_dict = {node: list(graph_from_edge_list.neighbors(node)) for node in graph_from_edge_list.nodes()}


population_size = 50
generations = 1000
mutation_rate = 0.1

start_time = time.time()
result, elapsed_time = evolve(graph_dict, population_size, generations, mutation_rate)

if result is not None:
    result_individual = result
    print("Hamiltonian cycle found:", result_individual)
    print(f"Elapsed time: {elapsed_time:.6f} seconds")
    draw_graph({result_individual[i]: graph_dict[result_individual[i]] for i in range(len(result_individual))}, title="Graph with Hamiltonian Cycle")
else:
    print("No Hamiltonian cycle found within the specified generations.")
