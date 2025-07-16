from TSP_tour import TSP_Tour

def read_input():
    # Read the input
    type_of_tour = input()
    N = int(input())
    coordinates = []
    graph = []
    for i in range(N):
        x, y = [float(j) for j in input().split(" ")]
        coordinates.append((x, y))
    for i in range(N):
        row = [float(j) for j in input().split(" ")]
        graph.append(row)
    
    return type_of_tour, N, coordinates, graph

def print_tour(n, tour):
    # Print the tour
    for i, city in enumerate(tour):
        if i == n-1: print(city)
        else: print(city, end=" ")

def main():
    type_of_tour, N, coordinates, graph = read_input()
    # Start time
    tsp = TSP_Tour(graph)
    lowest_cost = float('inf')
    
    # Find the tour using the greedy algorithm
    greedy_tour, greedy_cost = tsp.GreedyTour()
    #print("Greedy Cost: ", greedy_cost)
    if greedy_cost < lowest_cost:
        lowest_cost = greedy_cost
        print_tour(N, greedy_tour)
    
    # Find the tour using the nearest neighbor algorithm
    nearest_tour, nearest_cost = tsp.NearestNeighborTour()
    #print("Nearest Neighbor Cost: ", nearest_cost)
    if nearest_cost < lowest_cost:
        lowest_cost = nearest_cost
        print_tour(N, nearest_tour)
    
    # Find the tour using the savings heuristic
    savings_tour, savings_cost = tsp.SavingsTour()
    #print("Savings Cost: ", savings_cost)
    if savings_cost < lowest_cost:
        lowest_cost = savings_cost
        print_tour(N, savings_tour)
    

if __name__ == "__main__":
    main()