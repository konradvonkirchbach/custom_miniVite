#!/usr/bin/env python
#
# This is a simple testcase purely for testing the autotuner on permutations
#
# http://en.wikipedia.org/wiki/Travelling_salesman_problem
#

# from builtins import range
# import adddeps #fix sys.path

import argparse
import logging
import random
import subprocess
import pty
from signal import *
import os
import time
import socket
import numpy as np
import utils
import json
import sys

import opentuner
from opentuner.search.manipulator import (ConfigurationManipulator, GraphMappingParameter)
from opentuner.search.objective import MinimizeTime
from opentuner.measurement import MeasurementInterface
from opentuner.measurement.inputmanager import FixedInputManager
from opentuner.tuningrunmain import TuningRunMain
from simpleGraph import simpleGraph

parser = argparse.ArgumentParser(parents=opentuner.argparsers())
parser.add_argument('-n', '--nodes', type=int, help='number of nodes')
parser.add_argument('-s', '--sockets', type=int, help='number of sockets')
parser.add_argument('-pps', '--processes_per_socket', type=int, help='processes per socket')
parser.add_argument('--seed_file', default="not set", help='File containing the permutation of the seed')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold with which better solutions are accepted')
parser.add_argument('--mpi_time', default='06:00:00', help='Time parsed to the mpi application')
parser.add_argument('--application_name', default='./miniVite', help='Name of Application to start')
parser.add_argument('--communication_graph_file',
                    help='JSON-file containing the communication graph information for the'
                         'GraphMappingParameter of opentuner')
parser.add_argument('--application_inputs', help='Input for the application, parsed via the command line')
parser.add_argument('--logging_level', default='WARNING', help='Logging level for python logger')
parser.add_argument('--model_threshold', type=float, default=1.1,
                    help='The threshold ratio sum / sum_best if a benchmark'
                         ' should be performed')
parser.add_argument('--socket_factor', type=float, default=1,
                    help='The factor to scale the probability of a swap if two ranks '
                         'are on one socket')
parser.add_argument('--node_factor', type=float, default=1, help='The factor to scale the probability of a swap if two'
                                                                 'ranks are on one compute node')
parser.add_argument('--global_factor', type=float, default=1,
                    help='The factor to scale the probability of a swap if two'
                         'ranks are not on the same hardware entity')

BUFFER_SIZE = 524288
HIGHLIGHTER = ' ################# '


class Mapper(MeasurementInterface):
    HOST = '10.10.10.37'
    PORT = 13002
    log = logging.getLogger('Mapper')
    seen_configurations = {}

    def __init__(self, args):
        self.log.setLevel(level=args.logging_level)
        super(Mapper, self).__init__(args)
        self.processes = args.nodes * args.processes_per_socket * args.sockets
        self.n_nodes = args.nodes
        self.pps = args.processes_per_socket
        self.n_sockets = args.sockets
        self.seed_file = args.seed_file
        self.threshold = args.threshold
        self.best_result = np.inf
        self.sum = np.inf
        self.bottleneck = np.inf
        self.mpi_time = args.mpi_time
        self.graph_file = args.communication_graph_file
        self.model_threshold = args.model_threshold
        self.socket_factor = args.socket_factor
        self.node_factor = args.node_factor
        self.global_factor = args.global_factor
        self.G = simpleGraph(self.graph_file)

        # start the mpi process
        cmds = ["srun", "-m", "block:cyclic", '--cpu-freq', 'High', '--cpu-bind', 'core', "-N", str(self.n_nodes),
                "--ntasks-per-node", "{}".format(self.pps * self.n_sockets), '-p', 'q_staff_lowip_long',\
                "-t", self.mpi_time, './' + args.application_name] + args.application_inputs.split(' ')

        try:
            # Socket comm stuff
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind((self.HOST, self.PORT))
            s.listen()
            self.proc = subprocess.Popen(cmds, stdout=subprocess.DEVNULL)
            #self.proc = subprocess.Popen(cmds)
            self.con, self.addr = s.accept()
            self.log.info(HIGHLIGHTER + "Established connection" + HIGHLIGHTER)

            self.warm_up()
        except RuntimeError:
            print(RuntimeError)
            self.log.error("KILLED PROCESS!")
            self.proc.kill()
            self.con.close()
            sys.exit(1)

    def __del__(self):
        try:
            self.con.close()
            self.proc.kill()
        except:
            self.log.warning("Could not kill connection")


    def warm_up(self):
        p = [i for i in range(self.processes)]
        slave_input_perm = ' '.join([str(i) for i in p])
        slave_input_perm += '\n'
        slave_input_flag = "1" + "\n"

        for _ in range(100):
            self.log.debug('Warmup> Sending benchmark flag')
            self.con.send(slave_input_flag.encode('utf-8'))
            self.log.debug('Warmup> Sent benchmark flag')

            self.log.debug('Warmup> Sending permutation')
            self.con.send(slave_input_perm.encode('ascii'))
            self.log.debug('Warmup> Sent permutation')
            t = self.con.recv(BUFFER_SIZE)



    def seed_configurations(self):
        """
        Look for seed configuration permutation in file 'seed_permutation.txt'.
        If this file doesn't exist, proceed with a consecutive assignment of ranks
        to processes
        """
        """
        return[{'Perm':p}]
        return [{'Perm': list(range(self.processes)), 'Perm' : random.shuffle(list(range(self.processes))),
                 'Perm_reversed' : (list(range(self.processes))).reverse()}]
        """
        if self.seed_file != 'not set':
            seeds = []
            try:
                with open(self.seed_file, 'r') as f:
                    seeds = json.load(f)
                print(seeds)
                #return self.translate_permutation_to_config(p)
            except RuntimeError:
                pass
                #p = list(range(self.processes))
                #return self.translate_permutation_to_config(p)
            p = [-1 for _ in range(self.processes)]
            l_ppn = self.G.getNumVertices() // self.n_nodes
            for node in range(self.n_nodes):
                for core in range(l_ppn):
                    p[self.pps * self.n_sockets * node + core] = l_ppn * node + core
            seeds.append({'Perm': p})
            print('Appended Blocked')
            return seeds
        else:
            """Hyperplane Permutation 16x16"""
            # p = [0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 64, 80, 96, 112, 65, 81, 97, 113, 66, 82, 98, 114, 67, 83, 99, 115, 68, 84, 100, 116, 69, 85, 101, 117, 70, 86, 102, 118, 71, 87, 103, 119, 8, 24, 40, 56, 9, 25, 41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61, 14, 30, 46, 62, 15, 31, 47, 63, 72, 88, 104, 120, 73, 89, 105, 121, 74, 90, 106, 122, 75, 91, 107, 123, 76, 92, 108, 124, 77, 93, 109, 125, 78, 94, 110, 126, 79, 95, 111, 127, 128, 144, 160, 176, 129, 145, 161, 177, 130, 146, 162, 178, 131, 147, 163, 179, 132, 148, 164, 180, 133, 149, 165, 181, 134, 150, 166, 182, 135, 151, 167, 183, 192, 208, 224, 240, 193, 209, 225, 241, 194, 210, 226, 242, 195, 211, 227, 243, 196, 212, 228, 244, 197, 213, 229, 245, 198, 214, 230, 246, 199, 215, 231, 247, 136, 152, 168, 184, 137, 153, 169, 185, 138, 154, 170, 186, 139, 155, 171, 187, 140, 156, 172, 188, 141, 157, 173, 189, 142, 158, 174, 190, 143, 159, 175, 191, 200, 216, 232, 248, 201, 217, 233, 249, 202, 218, 234, 250, 203, 219, 235, 251, 204, 220, 236, 252, 205, 221, 237, 253, 206, 222, 238, 254, 207, 223, 239, 255]
            """Hyperplane Permutation 6x31"""
            # p = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162, 168, 174, 180, 1, 2, 7, 8, 13, 14, 19, 20, 25, 26, 31, 32, 37, 38, 43, 44, 49, 50, 55, 56, 61, 62, 67, 68, 73, 74, 79, 80, 85, 86, 91, 92, 97, 98, 103, 104, 109, 110, 115, 116, 121, 122, 127, 128, 133, 134, 139, 140, 145, 146, 151, 152, 157, 158, 163, 164, 169, 170, 175, 176, 181, 182, 3, 4, 9, 10, 15, 16, 21, 22, 27, 28, 33, 34, 39, 40, 45, 46, 51, 52, 57, 58, 63, 64, 69, 70, 75, 76, 81, 82, 87, 88, 93, 94, 99, 100, 105, 106, 111, 112, 117, 118, 123, 124, 129, 130, 135, 136, 141, 142, 147, 148, 153, 154, 159, 160, 165, 166, 171, 172, 177, 178, 183, 184, 5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89, 95, 101, 107, 113, 119, 125, 131, 137, 143, 149, 155, 161, 167, 173, 179, 185]
            # seed_hp = self.translate_permutation_to_config(p)
            """kd Tree Permutation 16x16"""
            # p = [0, 1, 2, 6, 12, 7, 8, 13, 14, 3, 4, 5, 9, 15, 10, 11, 16, 17, 18, 24, 19, 20, 25, 26, 30, 36, 31, 32, 37, 38, 21, 27, 22, 23, 28, 29, 33, 39, 34, 35, 40, 41, 42, 48, 43, 44, 49, 50, 54, 60, 55, 56, 61, 62, 45, 51, 46, 47, 52, 53, 57, 63, 58, 59, 64, 65, 66, 72, 67, 68, 73, 74, 78, 84, 79, 80, 85, 86, 69, 75, 70, 71, 76, 77, 81, 87, 82, 83, 88, 89, 90, 96, 91, 92, 97, 98, 102, 108, 103, 104, 109, 110, 93, 99, 94, 95, 100, 101, 105, 111, 106, 107, 112, 113, 114, 120, 115, 116, 121, 122, 126, 132, 127, 128, 133, 134, 117, 123, 118, 119, 124, 125, 129, 135, 130, 131, 136, 137, 138, 144, 139, 140, 145, 146, 150, 156, 151, 152, 157, 158, 141, 147, 142, 143, 148, 149, 153, 159, 154, 155, 160, 161, 162, 168, 163, 164, 169, 170, 174, 180, 175, 176, 181, 182, 165, 171, 166, 167, 172, 173, 177, 183, 178, 179, 184, 185]
            self.log.debug(HIGHLIGHTER + "entered seed" + HIGHLIGHTER)
            seeds = []
            p = [-1 for _ in range(self.processes)]
            l_ppn = self.G.getNumVertices() // self.n_nodes
            for node in range(self.n_nodes):
                for core in range(l_ppn):
                    p[self.pps * self.n_sockets * node + core] = l_ppn * node + core
            # bottleneck, sum = utils.countInterNodeCommunication(self.G, p, self.n_nodes, self.pps*self.n_sockets)
            # if sum < self.sum and sum > 0:
            #    self.bottleneck, self.sum = bottleneck, sum
            #    self.init_perm = p
            # self.seen_configurations[utils.hash_permutation(p, self.n_sockets, self.n_nodes, self.pps)] = np.inf
            seeds.append({'Perm': p})
            p = [-1 for _ in range(self.processes)]
            for v in range(self.G.getNumVertices()):
                p[v] = v
            # bottleneck, sum = utils.countInterNodeCommunication(self.G, p, self.n_nodes, self.pps*self.n_sockets)
            # if sum < self.sum and sum > 0:
            #    self.bottleneck, self.sum = bottleneck, sum
            #    self.init_perm = p
            # self.seen_configurations[utils.hash_permutation(p, self.n_sockets, self.n_nodes, self.pps)] = np.inf
            seeds.append({'Perm': p})
            self.init_perm = list(range(self.processes))
            """
            for _ in range(40):
                p = util.greedy_partition_grower(self.n_nodes, self.pps * self.n_sockets, self.stencil, self.dims)
                mapping = util.mapping_from_permutation(p, self.dims, [self.pps * self.n_sockets] * self.n_nodes)
                bottleneck, sum = util.analyze_matrix(mapping, self.stencil)
                if sum < self.sum:
                    self.bottleneck, self.sum = bottleneck, sum
                    self.init_perm = p
                self.seen_configurations[util.hash_permutation(p, self.n_sockets, self.n_nodes, self.pps)] = np.inf
                #seeds.append(self.translate_permutation_to_config(p))
                seeds.append({'Perm': p})
            # FIX ME. Want to return the actual runtime of best seen configuration
            #seed_block = self.translate_permutation_to_config(list(range(self.processes)))
            """
            self.log.info(HIGHLIGHTER + "end of seed" + HIGHLIGHTER)
            return seeds
        # """

    def eval_solution(self, t, p):
        eps = self.threshold
        self.log.info('log> t={}, best={}, ratio={}'.format(t, self.best_result, t / self.best_result))
        if t / self.best_result < eps:
            self.best_result = t
            self.bottleneck, self.sum = utils.countInterNodeCommunication(self.G, p, self.n_nodes,
                                                                          self.pps * self.n_sockets)
            if not self.sum:
                self.sum = 1
            return t
        else:
            return t + abs(t - self.best_result)

    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data
        p = cfg['Perm']
        t = self.run_benchmark(p)
        return opentuner.resultsdb.models.Result(time=t)

    def checkPermutation(self, p):
        seenProcesses = [False for _ in range(self.G.getNumVertices())]
        for i in p:
            if i < 0:
                continue
            else:
                seenProcesses[i] = True
        allSet = True
        for b in seenProcesses:
            allSet &= b
        return allSet

    def run_benchmark(self, p):
        self.log.debug('All Processes set in Permutation = {}'.format(self.checkPermutation(p)))
        bottleneck, sum = utils.countInterNodeCommunication(self.G, p, self.n_nodes, self.pps * self.n_sockets)
        if sum > self.model_threshold * self.sum:
            self.log.info("Skipped permutation. Sum={}, Best Sum={}".format(sum, self.sum))
            return sum * self.best_result

        hash_value = utils.hash_permutation(p, self.n_sockets, self.n_nodes, self.pps)
        if hash_value in self.seen_configurations:
            self.log.info("Skipped permutation. Found in hash_keys=" + hash_value)
            return self.seen_configurations[hash_value]

        try:
            self.log.debug('Sending benchmark flag')
            slave_input = "1" + "\n"
            self.con.send(slave_input.encode('utf-8'))
            self.log.debug('Sent benchmark flag')

            slave_input = ' '.join([str(i) for i in p])
            slave_input += '\n'
            # print('log> Slave input', slave_input, " length is ", len(slave_input.encode('ascii')))
            # print('log> Slave input length is ', len(slave_input.encode('ascii')))
            self.log.debug('Sending permutation')
            self.con.send(slave_input.encode('ascii'))
            self.log.debug('Sent permutation')

            t = self.con.recv(BUFFER_SIZE)
            self.log.debug('Received time')
            try:
                t = float(t.decode('utf-8'))
                self.seen_configurations[hash_value] = t
                t = self.eval_solution(t, p)
            except:
                t = np.inf
        except RuntimeError:
            print(RuntimeError)
            print(HIGHLIGHTER, "KILLED PROCESS!", HIGHLIGHTER)
            self.proc.kill()
            sys.exit(1)
        return t

    def get_permutation(self, config):
        permutation = [-1 for _ in range(self.processes)]
        rank_values = []

        for rank in range(self.processes):
            node_id = config["Node_" + str(rank)]
            socket_id = config["Socket_" + str(rank)]
            core_id = config["Core_" + str(rank)]
            rank_values.append((node_id, socket_id, core_id, rank))

        return [r for _, _, _, r in sorted(rank_values)]

    def translate_permutation_to_config(self, p):
        n_nodes = self.n_nodes
        n_sockets = self.n_sockets
        n_cores_per_socket = self.pps
        ppn = n_sockets * n_cores_per_socket

        seed_dict = {}

        for i, rank in enumerate(p):
            seed_dict["Node_" + str(rank)] = i // n_nodes
            seed_dict["Socket_" + str(rank)] = (i % ppn) // n_cores_per_socket
            seed_dict["Core_" + str(rank)] = (i % ppn) % n_cores_per_socket

        return seed_dict

    def manipulator(self):
        manipulator = ConfigurationManipulator()
        p = [-1 for _ in range(self.processes)]
        for i in range(self.G.getNumVertices()):
            p[i] = i
        manipulator.add_parameter(GraphMappingParameter('Perm', p,
                                                        self.n_nodes, self.pps * self.n_sockets, self.n_sockets,
                                                        self.graph_file, self.socket_factor,
                                                        self.node_factor, self.global_factor))
        return manipulator

    def save_final_config(self, configuration):
        slave_input = "0\n"
        self.con.send(slave_input.encode('utf-8'))
        with open('AllConfigurations.json', 'w') as jsonFile:
            json.dump(self.seen_configurations, jsonFile)
        print("Init = ", str(self.init_perm))
        print('Best found = ', utils.hash_permutation(configuration.data['Perm'],
                                                      self.n_sockets, self.n_nodes,
                                                      self.pps))
        # print(configuration.data)


if __name__ == '__main__':
    args = parser.parse_args()
    Mapper.main(args)
