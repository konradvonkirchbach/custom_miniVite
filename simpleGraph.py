import json
import logging

class simpleGraph():
    def __initializeGraph(self, graphData: dict):
        self.__graph = {}
        for key, value in graphData.items():
            neighbors = []
            for neighbor, weight in value.items():
                neighbors.append((int(neighbor), int(weight)))
            self.__graph[int(key)] = neighbors

    def __readInputFile(self, filename):
        with open(filename, 'r') as jsonFile:
            data = json.load(jsonFile)
        self.log.debug("JSON data: {}".format(data))
        self.__initializeGraph(data)

    def __init__(self, filename, loggingLevel="WARNING"):
        self.log = logging.getLogger("simpleGraph-logger")
        self.log.setLevel(level=loggingLevel)
        self.__readInputFile(filename)
        self.__numVertices = len(self.__graph.keys())
        self.log.info("Read in graph")

    def getNumVertices(self):
        return self.__numVertices

    def getVertexIDs(self):
        return list(self.__graph.keys())

    def getUnweightedNeighbors(self, rank):
        try:
            return [i for i, _ in self.__graph[rank]]
        except:
            return []

    def getWeightedNeighbors(self, rank):
        try:
            return self.__graph[rank]
        except:
            return []