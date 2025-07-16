class TSP_Tour():
    def __init__(self, graph):
        self.graph = graph
    
    def tourCost(self, tour):
        # Compute the cost of the tour
        cost = 0
        for i in range(len(tour) - 1):
            cost += self.graph[tour[i]][tour[i + 1]]
        cost += self.graph[tour[-1]][tour[0]]
        return cost
    
    def GreedyHeuristic(self, tour = []):
        # Return the shortest edge from the graph as long as the edge doesn't complete a cycle
        search = []
        if tour == []: search = [i for i in range(len(self.graph))]
        else: search = [tour[0], tour[-1]]
        shortest = max(max(self.graph))
        edge = []
        for i in search:
            for j in range(len(self.graph[i])):
                if self.graph[i][j] < shortest and i != j and j not in tour:
                    shortest = self.graph[i][j]
                    edge = [i, j]
        return edge
    
    def NearestNeighborHeuristic(self, state, visited = []):
        # return the nearest neighbor of the current state
        # State is city
        neighbors = self.graph[state]
        nearest = max(neighbors)
        index = -1
        for i in range(len(neighbors)):
            if neighbors[i] < nearest and i != state and i not in visited:
                nearest = neighbors[i]
                index = i
        return index
    
    def savingsHeuristic(self, depot, i, j):
        # Compute the savings heuristic
        return self.graph[depot][i] + self.graph[depot][j] - self.graph[i][j]
    
    def computeSavings(self, depot):
        # Compute the savings for each pair of cities
        savings = []
        for i in range(len(self.graph)):
            for j in range(i+1, len(self.graph[i])):
                if i == j or i == depot or j == depot: continue
                savings.append((self.graph[depot][i] + self.graph[depot][j] - self.graph[i][j], i, j))
        savings.sort(reverse = True, key = lambda x: x[0])
        return savings
    
    def GreedyTour(self):
        # Compute the tour using greedy algorithm
        tour = []
        firstEdge = self.GreedyHeuristic()
        tour.append(firstEdge[0])
        tour.append(firstEdge[1])
        while len(tour) < len(self.graph):
            edge = self.GreedyHeuristic(tour)
            if edge[0] == tour[-1]: tour.append(edge[1])
            else: tour.insert(0, edge[1])
        return tour, self.tourCost(tour)
    
    def NearestNeighborTour(self):
        # Compute the tour using nearest neighbor algorithm
        lowest_cost = float('inf')
        best_tour = []
        for start in range(len(self.graph)):
            tour = [start]
            current = start
            while len(tour) < len(self.graph):
                current = self.NearestNeighborHeuristic(current, tour)
                tour.append(current)
            cost = self.tourCost(tour)
            if cost < lowest_cost:
                lowest_cost = cost
                best_tour = tour
        return best_tour, lowest_cost
    
    def SavingsTour(self):
        # Compute the tour using savings heuristic
        lowest_cost = float('inf')
        best_tour = []
        for depot in range(len(self.graph)):
            savings = self.computeSavings(depot)
            i = 0
            while depot in [savings[i][1], savings[i][2]]: i += 1
            tour = [depot, savings[i][1], savings[i][2], depot]
            for j in range(i+1, len(savings)):
                saving = savings[j]
                if saving[1] in tour and saving[2] in tour: continue
                if saving[1] in tour:
                    index = tour.index(saving[1])
                    if tour[index-1] == depot: tour.insert(index, saving[2])
                    elif tour[index+1] == depot: tour.insert(index+1, saving[2])
                elif saving[2] in tour:
                    index = tour.index(saving[2])
                    if tour[index-1] == depot: tour.insert(index, saving[1])
                    elif tour[index+1] == depot: tour.insert(index+1, saving[1])
            
            cost = self.tourCost(tour)
            if cost < lowest_cost:
                lowest_cost = cost
                best_tour = tour
        best_tour = best_tour[:-1]
        return best_tour, lowest_cost
        