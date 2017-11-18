import itertools
import random
import numpy as np

def tsp_solver(dists):
    """
    Implementation of Held-Karp, an algorithm that solves the Traveling
    Salesman Problem using dynamic programming with memoization.
    Parameters:
        dists: distance matrix
    Returns:
        A tuple, (cost, path).
    """
    n = len(dists)

    # Maps each subset of the nodes to the cost to reach that subset, as well
    # as what node it passed before reaching this subset.
    # Node subsets are represented as set bits.
    C = {}

    # Set transition cost from initial state
    for k in range(1, n):
        C[(1 << k, k)] = (dists[0][k], 0)

    # Iterate subsets of increasing length and store intermediate results
    # in classic dynamic programming manner
    for subset_size in range(2, n):
        for subset in itertools.combinations(range(1, n), subset_size):
            # Set bits for all nodes in this subset
            bits = 0
            for bit in subset:
                bits |= 1 << bit

            # Find the lowest cost to get to this subset
            for k in subset:
                prev = bits & ~(1 << k)

                res = []
                for m in subset:
                    if m == 0 or m == k:
                        continue
                    res.append((C[(prev, m)][0] + dists[m][k], m))
                C[(bits, k)] = min(res)

    # We're interested in all bits but the least significant (the start state)
    bits = (2**n - 1) - 1

    # Calculate optimal cost
    res = []
    for k in range(1, n):
        res.append((C[(bits, k)][0] + dists[k][0], k))
    opt, parent = min(res)

    # Backtrack to find full path
    path = []
    for i in range(n - 1):
        path.append(parent)
        new_bits = bits & ~(1 << parent)
        _, parent = C[(bits, parent)]
        bits = new_bits

    # Add implicit start state
    path.append(0)

    return opt, list(reversed(path))


def generate_distances(n):
    dists = [[0] * n for i in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            dists[i][j] = dists[j][i] = random.randint(1, 99)

    print(dists)
    return dists

def add_random_row(dist):
    """

    :param dist: Input Matrix
    :return: Matrix with one additional row and column
    """
    n=len(dist)+1
    P = np.zeros((n, n))
    values = np.random.randint(1,99,len(dist)+1) #randomly generate last row and column
    values[n-1]=0
    for i in range(n):
        for j in range (n):
            if(i==n-1):
                P[i][j] = values[j]
            elif(j==n-1):
                P[i][j]= values[i]
            else:
                P[i][j]=dist[i][j]

    P = P.astype(int)
    return P

if __name__ == '__main__':

    arg = 10

    dists = generate_distances(int(arg))

    # Pretty-print the distance matrix
    for row in dists:
        print(''.join([str(n).rjust(3, ' ') for n in row]))

    print('')

    print(tsp_solver(dists))

    #Solve for 11*11
    dists2 = add_random_row(dists)
    print (dists2)
    print (tsp_solver(dists2))

    #Solve for 12*12
    dists3 = add_random_row(dists2)
    print (dists3)
    print (tsp_solver(dists3))

    #solve for 13*13
    dists4 = add_random_row(dists3)
    print (dists4)
    print (tsp_solver(dists4))







