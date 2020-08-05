//
// Created by konradvonkirchbach on 7/16/20.
//

#ifndef MINIFE_REF_SRC_AUTOTUNERCONNECTION_H
#define MINIFE_REF_SRC_AUTOTUNERCONNECTION_H

#include <mpi.h>

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <string>
#include <unistd.h>
#include <cstring>
#include <vector>

#define PRINT(rank, X) 	if (!rank) std::cout << "MPI TCP log> " << X << std::endl;

int openTunerConnectionInit() {
  int w_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
  int sock = -1, valread;
  PRINT(w_rank, "Entered TCP Socket connection")
  if (!w_rank) {
	/*
	//NODE INFORMATION
	char hostname[255];
	gethostname(hostname, 255);
	std::cout << "MPI log> HOSTNAME=" << hostname << std::endl;
	 */
	struct sockaddr_in serv_addr;
	if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
	  std::cerr << "Socket creation error" << std::endl;
	  return -1;
	}
	//std::cout << "MPI log> sock " << sock << " has type " << typeid(sock).name() << std::endl;

	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(13002);
	//Convert IPv4 and IPv6 addresses from text to binary form
	if (inet_pton(AF_INET, "10.10.10.37", &serv_addr.sin_addr) <= 0) {
	  std::cerr << "Invalid address/ Address not supported" << std::endl;
	  return -1;
	}
	if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
	  std::cerr << "Connection Failed" << std::endl;
	  return -1;
	}
  }
    PRINT(w_rank, "Finished TCP Socket connection")
  return sock;
}

template <typename T>
void openTunerSendTime(int Socket, T time) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PRINT(rank, "Entered Send time")
  std::string data = std::to_string(time);
  if (!rank) send(Socket, data.data(), data.size(), 0);
  PRINT(rank, "Finished time send")
}


void openTunerSignalRecv(int Socket, int& flag) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    PRINT(rank, "Entered Signal receive")
  if (!rank) {
    char buff[1024];
    for (int i = 0; i < 1024; i++) buff[i] = 'A';
    read(Socket, buff, 1024);
    flag = std::atoi(buff);
  }
  MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
  PRINT(rank, "Finished TCP Socket connection")
}


int openTunerGetNewRank(int sock) {
  int w_size, w_rank, new_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &w_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &w_rank);
  std::vector<int> permutation(w_size, -1);
  std::string str_perm = "";
  int BUFFER_SIZE =  524288;

  if (!w_rank) {
	char* buffer = new char[BUFFER_SIZE];
	//std::cout << "MPI log> Before Reading permutation " << std::endl;
	for (int i {0}; i < BUFFER_SIZE; i++) buffer[i] = 'A';
	//std::cout << "MPI log> Allocated buffer permutation " << std::endl;
	int local_buff_size = 0;
	int in_read = 0;
	char* recv_buff[2000];
	do {
	  in_read = read(sock, recv_buff, 2000);
	  //std::cout << "MPI log> In_read " << in_read << std::endl;
	  std::memcpy(&buffer[local_buff_size], recv_buff, in_read);
	  local_buff_size += in_read;
	  if (buffer[local_buff_size - 1] == '\n') break;
	} while(in_read > 0);
	buffer[local_buff_size] = '\0';
	//int in_read = read(sock, buffer, BUFFER_SIZE);
	//std::cout << "MPI log> After Reading permutation " << local_buff_size << std::endl;
	//std::cerr << "MPI log>permutation buffer = " << buffer << std::endl;
	std::string str_buff = buffer;
	//std::cout << str_buff << std::endl;
	char* pch = std::strtok(buffer, " ");
	int index = 0, tmp;
	while (pch != NULL) {
	  tmp = std::atoi(pch);
	  //std::cout << "MPI log> tmp = " << tmp << std::endl;
	  if (tmp == -1) {
	    permutation[index] = MPI_UNDEFINED;
	  } else {
	  	permutation[index] = tmp;
	  }
	  pch = strtok(NULL, " ");
	  index++;
	}

	delete[] buffer;
	//std::cout << "MPI log> Finished permutation" << std::endl;

	/*
	for (int i : permutation)
	  str_perm += std::to_string(i) + " ";
	std::cout << "MPI Log> Perm = " << str_perm << std::endl;
	 */
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Scatter(permutation.data(), 1, MPI_INT, &new_rank, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return new_rank;
}

#endif //MINIFE_REF_SRC_AUTOTUNERCONNECTION_H
