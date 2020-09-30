#!/usr/bin/env python
#
# This is a simple testcase purely for testing the autotuner on permutations
#
# http://en.wikipedia.org/wiki/Travelling_salesman_problem
#

import argparse
import logging
import random
import subprocess
import json
import os
import time
import utils
import sys
import numpy as np

import opentuner
from opentuner.search.manipulator import (ConfigurationManipulator,
                                          PermutationParameter,
                                          IntegerParameter,
                                          CartesianMappingParameter)
from opentuner.search.objective import MinimizeTime
from opentuner.measurement import MeasurementInterface
from opentuner.measurement.inputmanager import FixedInputManager
from opentuner.tuningrunmain import TuningRunMain

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument('--configuration_file', default="not set", help='File containing the permutation of the seed')
parser.add_argument('--application_name', help='Name of application')
parser.add_argument('--application_input', default='', help='Input for application to be called')
parser.add_argument('--logging_level', default='INFO', help='Logging level')

N_NODES = [2, 4, 8, 16, 32]
PPNS = [2, 8, 12, 16, 20, 24, 32]
ALL_PAIRS = [(n, p) for n in N_NODES for p in PPNS]
N = -1
PPN = -1
PERMUTATION = []
ALL_VALUES = {}

MPIRUN = '/opt/spack/spack_git/opt/spack/linux-debian10-skylake_avx512/gcc-8.3.0/openmpi-4.0.3-mdfojmxodeqqxcmdpfh2mbgerdg7egfy/bin/mpirun'
JOBNAME = 'AUTOTUNE'

class Mapper(MeasurementInterface):
    log = logging.getLogger('RankFile-Logger')
    def __init__(self, args):
        self.log.setLevel(level=args.logging_level)
        super(Mapper, self).__init__(args)
        self.benchmarkOutputs = []
        self.seenPermutations = set()
        self.processes = N * PPN
        self.n_nodes = N
        self.ppn = PPN
        self.application_name = args.application_name
        self.application_input = args.application_input
        self.stencil = utils.five_point_stencil()
        self.log.info('Running with N = {} and ppn = {}'.format(self.n_nodes, self.ppn))


    def seed_configurations(self):
        """
        Look for seed configuration permutation in file 'seed_permutation.txt'.
        If this file doesn't exist, proceed with a consecutive assignment of ranks 
        to processes
        """
        blocked = list(range(self.processes))
        round_robin = list(range(self.processes))
        rank = 0
        for l_PPN in range(PPN):
            for l_N in range(N):
                round_robin[l_N * PPN + l_PPN] = rank
                rank += 1
        global PERMUTATION
        seed = [{'Perm': blocked}, {'Perm': round_robin}, {'Perm': PERMUTATION}]
        return seed


    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data
        p = cfg['Perm']
        hashKey = utils.hash_permutation(p, 2, self.n_nodes, self.ppn // 2)
        if hashKey in self.seenPermutations:
            return opentuner.resultsdb.models.Result(time=np.inf)
        self.seenPermutations.add(hashKey)
        t = self.run_benchmark(p)
        try:
            grid = utils.mapping_from_permutation(p, utils.getCartDims(self.processes), [self.ppn]*self.n_nodes)
            bottleneck, total = utils.analyze_matrix(grid, self.stencil)
        except:
            bottleneck, total = -1, -1
        self.benchmarkOutputs.append((p, float(t), int(bottleneck), int(total)))
        return opentuner.resultsdb.models.Result(time=t)


    def terminateAllocation(self):
        proc = subprocess.Popen(['squeue'], stdout=subprocess.PIPE)
        output, err = proc.communicate()
        output = output.decode('ascii').split('\n')
        for line in output:
            if JOBNAME in line:
                line = line.split(' ')
                line = [word for word in line if not word == '']
                jobid = line[0]
                subprocess.call(['scancel', jobid])
                break


    def run_benchmark(self, p):
        self.translate_permutation_to_rankfile(p)
        startingNodeId = 37 - self.n_nodes
        if startingNodeId < 10:
            startingNode = '0{}'.format(startingNodeId)
        else:
            startingNode = '{}'.format(startingNodeId)

        self.log.debug('Into run benchmark')
        allocateCmds = ['salloc', '-N', '{}'.format(self.n_nodes), '--ntasks-per-node', '{}'.format(self.ppn),
                        '--nodelist', 'hydra[{}-36]'.format(startingNode),
                        '--cpu-freq', 'High', '-t', '10:00', '-J', JOBNAME,
                        '-p', 'q_staff_low', '-m', 'block:cyclic', '--no-shell']

        allocateJob = False
        while not allocateJob:
            try:
                self.log.debug('Running {}'.format(allocateCmds))
                #Allocate nodes on Hydra
                allocation = subprocess.Popen(allocateCmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self.log.debug('Allocated nodes')
                allocateJob = True
            except:
                allocateJob = False

        runCmds = [MPIRUN, '-np', '{}'.format(self.processes), '--rankfile',
                   '/home/staff/vonkirchbach/cartesian_reordering/rank_file.txt',
                   self.application_name, self.application_input]

        runJob = False
        while not runJob:
            try:
                self.log.info('Running {}'.format(runCmds))
                run = subprocess.Popen(' '.join(runCmds), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, err = run.communicate()
                output = output.decode('ascii').split(' ')
                time = float(output[-2])
                self.log.debug('Executed MPI process')
                runJob = True
            except:
                print(output, err)
                runJob = False

        #Exit allocation
        self.terminateAllocation()
        self.log.debug('Deallocated nodes')

        return time


    def translate_permutation_to_rankfile(self, p):
        n_nodes = self.n_nodes
        n_cores_per_socket = self.ppn // 2
        starting_node = 37 - n_nodes

        rankDict = {}

        for core, rank in enumerate(p):
            rankDict[rank] = core

        with open("rank_file.txt", "w") as f:
            for rank in range(len(p)):
                i = rankDict[rank]
                node_id = starting_node + i // self.ppn
                socket_id = i % self.ppn // n_cores_per_socket
                core_id = (i % self.ppn) % n_cores_per_socket
                f.write("rank {}=hydra{} slot={}:{}\n".format(rank, node_id, socket_id, core_id))
        self.log.debug('Wrote ranke file')


    def manipulator(self):
        manipulator = ConfigurationManipulator()

        manipulator.add_parameter(CartesianMappingParameter('Perm', list(range(self.processes)),
                                                self.stencil,
                                                self.n_nodes, 
                                                self.ppn,
                                                2,
                                                utils.getCartDims(self.processes),
                                                1, 1, 1))
        return manipulator


    def save_final_config(self, configuration):
        global ALL_VALUES
        ALL_VALUES['{} {}'.format(int(self.n_nodes), int(self.ppn))] = self.benchmarkOutputs
        print("Best found = {}".format(configuration.data))


def getCartMappings(jsonFilePath):
    with open(jsonFilePath, 'r') as f:
        data = json.load(f)
    return data


def main(args):
    CART_MAPPINGS = getCartMappings(args.configuration_file)
    args = parser.parse_args()

    for hardwareConfig, permutation in CART_MAPPINGS.items():
        global N
        global PPN
        global PERMUTATION
        N, PPN = hardwareConfig.split(':')
        N, PPN = int(N), int(PPN)
        PERMUTATION = permutation
        Mapper.main(args)

    with open('allConfigurationsOutput.csv', 'w') as f:
        f.write('Number of Nodes, Processes per Node, Permutation, Time, Bottleneck, Total\n')
        for key, val in ALL_VALUES.items():
            keyItems = key.split(' ')
            for data in val:
                data = [str(i) for i in data]
            f.write(','.join(keyItems + data) + '\n')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)