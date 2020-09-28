"""
Utils for opentuner mapper
author: Konrad von Kirchbach
"""
import numpy as np
import random
import logging
from simpleGraph import simpleGraph
import math


def translate_permutation_to_rankfile(p, n_nodes, ppn, filename='RankFile.txt'):
    n_cores_per_socket = ppn // 2
    starting_node = 37 - n_nodes

    rankDict = {}

    for core, rank in enumerate(p):
        rankDict[rank] = core

    with open(filename, "w") as f:
        for rank in range(len(p)):
            i = rankDict[rank]
            node_id = starting_node + i // ppn
            socket_id = i % ppn // n_cores_per_socket
            core_id = (i % ppn) % n_cores_per_socket
            hydraName = 'hydra'
            if node_id < 10:
                hydraName += '0' + str(node_id)
            else:
                hydraName += str(node_id)
            f.write("rank {}={} slot={}:{}\n".format(rank, hydraName, socket_id, core_id))
    print('Wrote ranke file')


def getPrimes(n):
    primes = []
    # Print the number of two's that divide n
    while n % 2 == 0:
        primes.append(2)
        n = n / 2

    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        i = int(i)
        # while i divides n , print i ad divide n
        while n % i == 0:
            primes.append(i)
            n = n / i

    # Condition if n is a prime
    # number greater than 2
    if n > 2:
        primes.append(int(n))

    return primes

def getCartDims(p, ndims=2):
    primes = getPrimes(p)
    dims = [1 for _ in range(ndims)]
    for prime in sorted(primes, reverse=True):
        minimum = min(dims)
        index = dims.index(minimum)
        dims[index] *= prime
    dims.sort(reverse=True)
    return dims


def getNumberOfNodesUsed(p):
    """
    Given a permutation p, with possible unassigned cores, denoted with -1,
    extract the number of compute nodes and processes per node used, by counting
    the number of non-minus-ones on the first node to get the processes per node
    and the divide length of p by computed processes per node. Assumes that the
    ranks on the first node are the first entries in p
    :param p: permutation list of ranks with possible unassigned processes (denoted with -1)
    :return: number of used compute nodes and processes per node
    """
    ppn = 0
    index = 0
    #skip the -1 ones
    while index < len(p):
        if p[index] >= 0:
            break
        index += 1
    #count processes on first node
    while index < len(p):
        if p[index] < 0:
            break
        ppn += 1
        index += 1
    totalP = ppn
    #count total processes
    while index < len(p):
        if p[index] >= 0:
            totalP += 1
        index += 1
    n_nodes = math.ceil(totalP / ppn)
    return (n_nodes, ppn)


def countInterNodeCommunication(G, p, g_nodes, g_ppn):
    """
    Simple method that collects the number of internode communication, both sum and bottleneck
    for a permutation p, given n_nodes number of nodes with ppn processes per node. The graph is
    assumed to be directed
    :param G: simpleGraph needed for neighbor extraction
    :param p: permutation parameter as list
    :return: a tuple with bottleneck b and total s number of inter-node connections b,s
    """
    #n_nodes, ppn = getNumberOfNodesUsed(p)
    #bottleneck, sum
    sum = 0
    perNode = [0 for _ in range(g_nodes)]
    for index, rank in enumerate(p):
        # Takes care of the possibility that we have more hardware ressources than processes
        if rank < 0:
            continue
        rankNodeID = index // g_ppn
        for neighbor in G.getUnweightedNeighbors(rank):
            neighborIndex = p.index(neighbor)
            neighborNodeID = neighborIndex // g_ppn
            if neighborNodeID != rankNodeID:
                sum += 1
                perNode[rankNodeID] += 1
        index += 1
    return (max(perNode), sum)


def hash_permutation(p: [int], n_sockets: int, n_nodes: int,  pps: int):
    """
    Takes a permutation array and produces a hash key in string form.
    Ranks on a socket a sorted and delimited with a '_', sockets are
    delimited by '__', whereas nodes are delimited with '-'.
    :param p: permutation array
    :param n_sockets: number of sockets
    :param pps: processes per socket
    :return: string array to be cashed
    """
    ppn = n_sockets * pps
    node_ranks = [p[ppn*i:ppn+ppn*i] for i in range(n_nodes)]
    socket_ranks = [[sorted(node[i*pps:i*pps + pps]) for i in range(n_sockets)] for node in node_ranks]
    socket_ranks.sort()
    return str(socket_ranks)


def mapping_from_permutation(permutation: [int], dims: [int], partitions: [int]) -> np.ndarray:
    grid = np.zeros(tuple(dims), np.int32)
    for i, new_rank in enumerate(permutation):
        new_coord = np.unravel_index(new_rank, tuple(dims))
        grid[new_coord] = color_for_index(i, partitions)

    return grid


def color_for_index(index: int, partitions: [int]) -> int:
    color = 0
    while index > 0:
        if index < partitions[color]:
            break
        else:
            index -= partitions[color]
            color += 1
    return color


def analyze_matrix(grid: np.ndarray, stencil: [tuple]):
    edges_per_part = dict()
    it = np.nditer(grid, flags=['multi_index'])
    dim_sizes = grid.shape
    while not it.finished:
        num_outgoing = 0
        coord = it.multi_index
        # print('coord ', coord, ' has neighbours:')
        for vector in stencil:
            target_coord = np.add(coord, vector)
            missed = False
            for dim in range(grid.ndim):
                if target_coord[dim] < 0 or target_coord[dim] >= dim_sizes[dim]:
                    missed = True
                    break
            if missed:
                continue
            target_partition = grid[tuple(target_coord)]
            # print(' ', target_coord, ' with color ', target_partition)
            if target_partition != it[0]:
                num_outgoing += 1
        # print('  coord ', coord, ' has ', num_outgoing, ' outgoing edges')
        own_color = grid[tuple(coord)]
        if own_color not in edges_per_part:
            edges_per_part[own_color] = num_outgoing
        else:
            edges_per_part[own_color] += num_outgoing
        it.iternext()
    sum = 0
    bottleneck = -1
    for k, v in edges_per_part.items():
        if v > bottleneck:
            bottleneck = v
        sum += v
    # print(edges_per_part)

    return bottleneck, sum


def five_point_stencil(ndims=2) -> list:
    stencil_size = 2*ndims**2
    n_neighbors = 2*ndims
    stencil = [0]*stencil_size

    dim_index = 0
    for i in range(0, stencil_size, n_neighbors):
        stencil[i + dim_index] = 1
        stencil[i + dim_index + ndims] = -1
        dim_index += 1

    return stencil_converter(stencil, ndims)


def stencil_converter(stencil: list, ndims: int) -> list:
    """
    Takes in the flattend stencil list and returns it
    in forms of tuples
    :param stencil: flattend list of rel. coord. Ex: [0, 1, 0, 2], ndims = 2
    :return: list with tuples. Ex: [(0, 1), (0, 2)]
    """
    converted_stencil = []
    for i in range(0, len(stencil), ndims):
        new_tuple = []
        for d in range(ndims):
            new_tuple.append(stencil[i + d])
        converted_stencil.append(tuple(new_tuple))
    return converted_stencil


def extract_neighbors(rank: int, stencil: list, dims: list) -> list:
    """
    Given a rank on a grid with dimensions dims, extract all the neighbor ranks
    on the grid, assuming non-periodicity
    :param rank: Rank of interest
    :param stencil: List of tuples of integer describing offset
    :param dims: Integer dimensions of grid
    :return: Array of neighbor ranks
    """
    neighbors = []
    rank_coord = np.unravel_index(rank, dims)
    #logging.info("Calling rank {}, rank coord {}, dims {}, stencil {}".format(rank, rank_coord, dims, stencil))
    d = len(dims)

    #Expect flatted list, i.e., len(stencil)/len(dims is number of neighbors
    for relativ_coord in stencil:
        neighbor_coord = [i for i in rank_coord]
        inside_grid = True

        for idx, offset in enumerate(relativ_coord):
            neighbor_coord[idx] += offset
            #Neighbor not inside of grid. Don't need to bother
            #logging.info("Neighbor coord={}".format(neighbor_coord))
            if neighbor_coord[idx] >= dims[idx] or neighbor_coord[idx] < 0:
                inside_grid = False

        if inside_grid:
            neighbors.append(np.ravel_multi_index(neighbor_coord, dims))

    return neighbors


def greedy_partition_grower(n_nodes: int, ppn: int, stencil: list, grid: list) -> list:
    """
    For each partition, randomly select a starting vertex and greedily expand the graph
    around it.
    :param n_node: Number of nodes
    :param ppn: Number of processes per node
    :param stencil: integer flattened list describing offset
    :param grid: integer list describing the dimensions of the grid
    :return: p = permutation list
    """
    n_proc = n_nodes * ppn
    unassigned_vertices = list(range(n_proc))
    p = [-1] * n_proc
    for node_idx in range(n_nodes):
        boundary_vertices = []
        start_index = random.choice(range(len(unassigned_vertices)))
        start_rank = unassigned_vertices.pop(start_index)
        p[node_idx * ppn] = start_rank
        boundary_vertices += extract_neighbors(start_rank, stencil, grid)
        assigned_to_node = 1

        # Select a random vertex from boundary region and remove it from list
        while assigned_to_node < ppn:
            if len(boundary_vertices):
                next_rank = boundary_vertices.pop(random.choice(range(len(boundary_vertices))))
            else:
                next_rank = unassigned_vertices.pop(random.choice(range(len(unassigned_vertices))))
            #Check if it assigned in p and assign if not
            if not next_rank in p:
                p[node_idx * ppn + assigned_to_node] = next_rank
                #logging.info("Next rank {}".format(next_rank))
                boundary_vertices += extract_neighbors(next_rank, stencil, grid)
                assigned_to_node += 1
            #Remove it from unassigned vertices
            try:
                unassigned_vertices.remove(next_rank)
            except:
                pass

    return p