#include "align.h"
#include "comm.h"
#include "emulator.h"
#include <assert.h>
#include <cinttypes>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <string>
using namespace std;

ncclResult_t TopologyInit(EmuTopology *topology, ncclProxyOp *proxyOp,
                             ncclInfo *info) {
  LOG_NOTE(NCCL_NOTE, "TopologyInit kbypass=%d", KERNEL_BYPASS);
  if (KERNEL_BYPASS == 1) {
    ncclComm *comm = info->comm;
    int nranks = comm->nRanks;
    int myrank = comm->rank;
    int nchannels = info->nChannels;
    LOG_NOTE(NCCL_NOTE,
            "TopologyInit for rank: %d, ringmapsize=%lu, inited:%d", myrank,
            topology->ringmap.size(), topology->init);
    if ((topology->init & topoInitState::META_INITED) == 0) {

      topology->nranks = nranks;
      topology->nnodes = NUM_NODES; // should be set by application!
      topology->nrankpernode = topology->nranks / topology->nnodes;
      assert(topology->nranks % topology->nnodes == 0);
      topology->init =
          (topoInitState)(topology->init | topoInitState::META_INITED);
    }
    if ((topology->init & topoInitState::PER_CALL_INITED) == 0) {
      topology->nchannels = nchannels;
      topology->myranks.clear();
      topology->init =
          (topoInitState)(topology->init | topoInitState::PER_CALL_INITED);
      LOG_NOTE(NCCL_NOTE, "Myranks cleared!");
    }
    topology->myranks.insert(myrank);
  }
  return ncclSuccess;
}

ncclResult_t TopologyUpdateMap(EmuTopology *topology, int rank, int channel,
                                  ncclRing *ring, int *ringranks, int nranks) {
  topology->prev[rank] = ring->prev;
  topology->next[rank] = ring->next;
  topology->ringmap[make_pair(rank, channel)] = ring->index;

  if (rank == 0) {
    for (int i = 1; i < nranks; ++i) {
      topology->ringmap[make_pair(i, channel)] = ringranks[i];
      LOG_NOTE(NCCL_NOTE, "update ringmap rk%d ringidx%d ch%d", i,
              topology->ringmap[make_pair(i, channel)], channel);
    }
  }
  LOG_NOTE(NCCL_NOTE, "TopologyUpdateMap [%d->%d->%d], mapsize=%lu",
          topology->prev[rank], rank, topology->next[rank],
          topology->ringmap.size());
  return ncclSuccess;
}

ncclResult_t TopologyDestroy(EmuTopology *topology) {
  assert(topology->init & topoInitState::PER_CALL_INITED);
  topology->init =
      (topoInitState)(topology->init ^ topoInitState::PER_CALL_INITED);
  LOG_NOTE(NCCL_NOTE, "TopologyDestroy");
  return ncclSuccess;
}