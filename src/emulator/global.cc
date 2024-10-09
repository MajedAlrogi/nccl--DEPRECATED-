#include "align.h"
#include "comm.h"
#include "emulator.h"
#include <assert.h>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <string>
using namespace std;

int KERNEL_BYPASS = -1;
int NUM_NODES = -1;
int MY_NODE = -1;
Coordinator global_coordinator;
EmuTopology global_topology;
mutex emulator_lock;

ncclResult_t getAllEnvVars() {
  setbuf(stdout, NULL);
  setbuf(stderr, NULL);
  LOG_NOTE(NCCL_NOTE, "getAllEnvVars");
  char *env = getenv("OMPI_COMM_WORLD_SIZE");
  if (env == NULL) {
    env = getenv("MOD_N_MPI_RANKS");
  }
  if (env == NULL) {
    LOG_NOTE(NCCL_LOG_ABORT, "Error: N_MPI_RANKS not set");
    return ncclModError;
  } else {
    NUM_NODES = atoi(env);
    LOG_NOTE(NCCL_NOTE, "MOD_N_MPI_RANKS=%d", NUM_NODES);
  }
  env = getenv("OMPI_COMM_WORLD_RANK");
  if (env == NULL) {
    env = getenv("MOD_MY_MPI_RANK");
  }
  if (env == NULL) {
    LOG_NOTE(NCCL_LOG_ABORT, "Error: MY_MPI_RANK not set");
    return ncclModError; // TODO: ALI Alfarhan check the logging
  } else {
    MY_NODE = atoi(env);
    LOG_NOTE(NCCL_NOTE, "MY_MPI_RANK=%d", MY_NODE);
  }
  env = getenv("MOD_KERNEL_BYPASS");
  if (env == NULL) {
    LOG_NOTE(NCCL_NOTE, "MOD_KERNEL_BYPASS not set, default to 0");
    KERNEL_BYPASS = 0;
  } else {
    KERNEL_BYPASS = atoi(env);
    LOG_NOTE(NCCL_NOTE, "MOD_KERNEL_BYPASS=%d", KERNEL_BYPASS);
  }
  return ncclSuccess;
}