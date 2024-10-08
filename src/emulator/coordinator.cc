#include "align.h"
#include "comm.h"
#include "driver_types.h"
#include "emulator.h"
#include <assert.h>
#include <cinttypes>
#include <map>
#include <math.h>
#include <stdlib.h>
#include <string>
using namespace std;



/** 
 * This file has the following objectives:
 * 1. Initialize the coordinator and collective operations
 * 2. Simulate communication WITHOUT GPU computation
 * 3. ensure that data is transmitted in the correct order and that collective communication operations follow the appropriate workflow. 
 *    It manages the send and receive progress (tracked by sendtail and recvtail) for each channel and rank, 
 *    ensuring that messages are sent and received in the correct order, even in the absence of real computation.
 * 
*/


/**
 * SECTION 1: HELPER METHODS
 * 
 * This section contains 2 main parts:
 * 
 * Part A: Sending/Recieving helper methods
 * Part B: Synchronization helper methods
*/

// Part A: Sending/Recieving helper methods
static void calc_size_inkernel(int nelem, vector<int> &res) {
  LOG_MOD(NCCL_MOD, "calc_size_inkernel: nelem=%d", nelem);
  int stepSize = 131072; // DEFAULT_BUFFSIZE(simple) / NCCL_STEP / sizeof(float)
  int SlicePerChunk = 2; // all reduce
  int StepPerSlice = 2;  //! i don't know why
  int sliceSize = stepSize * StepPerSlice;
  sliceSize = std::max(DIVUP(nelem, 16 * SlicePerChunk) * 16, sliceSize / 32);
  int offset = 0, slice = 0;
  while (offset < nelem) {
    int size = sliceSize < nelem - offset ? sliceSize : nelem - offset;
    res.push_back(size);
    offset += sliceSize;
    slice += 1;
  }
  while (slice < SlicePerChunk) {
    sliceSize = sliceSize < nelem - offset ? sliceSize : nelem - offset;
    res.push_back(0);
    offset += sliceSize;
    slice += 1;
  }
}

// calculate the expected send size for myrank
// this is also used to calculated the recv size for the rank that will
// receive from myrank
static void calc_size_channel(int nranks, int ringindex, int count,
                              int nchannels, int mychannel, int nthreads,
                              int tsize, vector<int> &res) {
  const int chunkSize = 524288;
  int bid = mychannel;
  int loopSize = nchannels * nranks * chunkSize;
  int size = count;
  int ringIx = ringindex;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    ssize_t realChunkSize;
    // if proto == Simple
    realChunkSize =
        min(chunkSize, (int)DIVUP(size - gridOffset, nchannels * nranks));
    realChunkSize =
        ROUNDUP(realChunkSize, (nthreads - 32) * sizeof(uint64_t) / tsize);
    realChunkSize = int(realChunkSize);

    LOG_MOD(NCCL_MOD, "realChunkSize=%lu, nthreads=%d", realChunkSize,
            nthreads);

    auto calcOffset = [&](int chunk) -> ssize_t {
      return gridOffset + bid * nranks * realChunkSize + chunk * realChunkSize;
    };
    auto modRanks = [&](int r) -> int {
      return r - (r >= nranks ? nranks : 0);
    };

    ssize_t offset;
    int nelem;
    int chunk;

    // step 0: push data to next GPU
    chunk = modRanks(ringIx + nranks - 1);
    offset = calcOffset(chunk);
    nelem = std::min(realChunkSize, size - offset);
    calc_size_inkernel(nelem, res);

    // k-2 steps: reduce and copy to next GPU
    for (int j = 2; j < nranks; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      offset = calcOffset(chunk);
      nelem = std::min(realChunkSize, size - offset);
      calc_size_inkernel(nelem, res);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ringIx + 0;
    offset = calcOffset(chunk);
    nelem = std::min(realChunkSize, size - offset);
    calc_size_inkernel(nelem, res);

    // k-2 steps: copy to next GPU
    for (int j = 1; j < nranks - 1; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      offset = calcOffset(chunk);
      nelem = std::min(realChunkSize, size - offset);
      calc_size_inkernel(nelem, res);
    }
  }
  for (int i = 0; i < res.size(); i++) {
    res[i] *= tsize;
  }
}

static void calc_sendsize_channel(int nranks, int myrank, int count,
                                  int nchannels, int mychannel, int nthreads,
                                  int tsize, vector<int> &res) {
  auto &ringmap = global_topology.ringmap;
  assert(ringmap.count(make_pair(myrank, mychannel)) > 0);
  int myringix = ringmap[make_pair(myrank, mychannel)];
  calc_size_channel(nranks, myringix, count, nchannels, mychannel, nthreads,
                    tsize, res);
  std::string szs;
  for (int i = 0; i < res.size(); ++i) {
    szs = szs + " " + std::to_string(res[i]);
  }
  LOG_MOD(NCCL_MOD, "Calculated send sizes for ringix:%d rank %d: %s, ch=%d",
          myringix, myrank, szs.c_str(), mychannel);
}

static void calc_recvsize_channel(int nranks, int myrank, int count,
                                  int nchannels, int mychannel, int nthreads,
                                  int tsize, vector<int> &res) {
  auto &ringmap = global_topology.ringmap;
  assert(ringmap.count(make_pair(myrank, mychannel)) > 0);
  int target = global_topology.prev[myrank];
  int target_ringix = ringmap[make_pair(target, mychannel)];
  calc_size_channel(nranks, target_ringix, count, nchannels, mychannel,
                    nthreads, tsize, res);
  std::string szs;
  for (int i = 0; i < res.size(); ++i) {
    szs = szs + " " + std::to_string(res[i]);
  }
  LOG_MOD(NCCL_MOD,
          "Calculated recv sizes for ringix:%d targetrk:%d, rank %d: %s, ch=%d",
          target_ringix, target, myrank, szs.c_str(), mychannel);
}

// Part B: Synchronization helper methods
static int check_done_ch(ChannelInfo &ch) {
  if (ch.sendtail == ch.sendsizes.size() &&
      ch.recvtail == ch.recvsizes.size()) {
    return 1;
  }
  return 0;
}

static int update_done_rank(RankInfo &rank) {
  if (rank.done) {
    return 1;
  }
  int done = 1;
  for (int i = 0; i < rank.channels.size(); ++i) {
    done = done & check_done_ch(rank.channels[i]);
  }
  rank.done = done;
  LOG_MOD(NCCL_MOD, "rank update check_done_rank: done=%d, rank=%d", done,
          rank.myrank);
  return done;
}

static void update_done(Coordinator *coordinator) {
  if (coordinator->done) {
    return;
  }
  int done = 1;
  for (int i = 0; i < coordinator->ranks.size(); ++i) {
    done = done & update_done_rank(coordinator->ranks[i]);
  }
  if (done) {
    coordinator->done = 1;
    LOG_MOD(NCCL_MOD, "coordinator update check_done: done");
  }
}

/**
 * SECTION 2: Coordinator Initialization
 * The main functions here are: 
 * >> coordinatorInit
 * >> coordinatorDestroy
 * 
 * CoordinatorInit: The main initialization function for the coordinator. It calls metaInit to set up the task metadata, 
 * then uses sendrecvInit to initialize the sending and receiving ranks and their respective channels. 
 * This function essentially sets up the entire communication framework for the task.
 * 
 * modCoordinatorDestroy: Resets and cleans up the coordinator once the task is completed, 
 * ensuring that resources are released, and the system can be prepared for the next task.
*/

static void rankInit(Coordinator *coordinator, int rank) {
  int nchannels = coordinator->task.nchannels;
  int nthreads = coordinator->task.nthreads;
  int nranks = coordinator->comm.nranks;
  int count = coordinator->task.count;

  RankInfo &rankinfo = coordinator->ranks[rank];
  rankinfo.myrank = rank;
  rankinfo.send = 0;
  rankinfo.recv = 0;
  if (rankinfo.myrank == coordinator->sendrank) {
    rankinfo.send = 1;
  }
  if (rankinfo.myrank == coordinator->recvrank) {
    rankinfo.recv = 1;
  }
  LOG_MOD(NCCL_MOD, "rankInit: myrank=%d, send=%d, recv=%d", rankinfo.myrank,
          rankinfo.send, rankinfo.recv);
  rankinfo.channels = vector<ChannelInfo>();
  for (int i = 0; i < nchannels; ++i) {
    ChannelInfo ch;
    ch.bid = i;
    ch.sendsizes = vector<int>();
    ch.recvsizes = vector<int>();
    ch.send = rankinfo.send;
    ch.recv = rankinfo.recv;
    if (rankinfo.send) {
      calc_sendsize_channel(nranks, rankinfo.myrank, count, nchannels, i,
                            nthreads, sizeof(float), ch.sendsizes);
    }
    if (rankinfo.recv) {
      calc_recvsize_channel(nranks, rankinfo.myrank, count, nchannels, i,
                            nthreads, sizeof(float), ch.recvsizes);
    }
    ch.sendtail = 0;
    ch.recvtail = 0;
    rankinfo.channels.push_back(ch);
  }
}

static void metaInit(Coordinator *coordinator, ncclProxyOp *proxyOp,
                     ncclInfo *info) {
  if (!coordinator->init) {
    coordinator->init = 1;
    coordinator->done = 0;

    delete coordinator->proxyOp;
    coordinator->proxyOp = new ncclProxyOp;
    *coordinator->proxyOp = *proxyOp;
    delete coordinator->info;
    coordinator->info = new ncclInfo;
    *coordinator->info = *info;

    TaskInfo task;
    task.count = info->count;
    task.tsize = sizeof(float);
    task.coll = 0;
    task.reduceOp = 0;
    task.algo = 0;
    task.nchannels = info->nChannels;
    task.nthreads = info->nThreads;

    CommInfo comm;
    comm.nranks = info->comm->nRanks;
    comm.nnodes = NUM_NODES; // should be set by application!
    comm.mynode = MY_NODE; // should be set by application!
    comm.nrankpernode = comm.nranks / comm.nnodes;
    assert(comm.nranks % comm.nnodes == 0);

    coordinator->comm = comm;
    coordinator->task = task;
    coordinator->ranks = map<int, RankInfo>();
  }
}

static void sendrecvInit(Coordinator *coordinator, EmuTopology *topology) {
  LOG_MOD(NCCL_MOD, "sendrecvInit, myranks.size=%lu, nrankpernode=%d",
          topology->myranks.size(), topology->nrankpernode);
  if (topology->myranks.size() < topology->nrankpernode) {
    return;
  }
  map<int, bool> ismynode;

  for (int i = 0; i < topology->nranks; ++i) {
    ismynode[i] = false;
  }
  for (auto i : topology->myranks) {
    ismynode[i] = true;
  }
  coordinator->sendrank = -1;
  coordinator->recvrank = -1;
  for (auto i : topology->myranks) {
    auto prev = topology->prev[i];
    auto next = topology->next[i];
    LOG_MOD(NCCL_MOD, "rank=%d, prev=%d, next=%d, ismynode[rank]=%d", i, prev,
            next, (int)ismynode[i]);
    if (!ismynode[next]) {
      assert(coordinator->sendrank == -1);
      coordinator->sendrank = i;
    }
    if (!ismynode[prev]) {
      assert(coordinator->recvrank == -1);
      coordinator->recvrank = i;
    }
  }
  if (coordinator->sendrank != -1 && coordinator->recvrank != -1) {
    LOG_MOD(
        NCCL_MOD, "sendrecv solved: sendrank=%d, recvrank=%d, ringmapsize=%lu",
        coordinator->sendrank, coordinator->recvrank, topology->ringmap.size());
    for (auto i : topology->myranks) {
      rankInit(coordinator, i);
    }
  }
}

ncclResult_t coordinatorInit(Coordinator *coordinator,
                                ncclProxyOp *proxyOp, ncclInfo *info) {
  LOG_MOD(NCCL_MOD, "CoordinatorInit kbypass=%d", KERNEL_BYPASS);
  if (KERNEL_BYPASS == 1) {
    metaInit(coordinator, proxyOp, info);
    int count = coordinator->task.count;
    assert(count == info->count);
    ncclComm *comm = info->comm;
    int nranks = comm->nRanks;
    int myrank = comm->rank;
    int nchannels = info->nChannels;
    int nthreads = info->nThreads;
    LOG_MOD(NCCL_MOD,
            "CoordinatorInit: kbypass=%d, count=%d, nranks=%d, myrank=%d, "
            "nchannels=%d, "
            "nthreads=%d",
            KERNEL_BYPASS, count, nranks, myrank, nchannels, nthreads);
    sendrecvInit(coordinator, &global_topology);
  }
  return ncclSuccess;
}

ncclResult_t coordinatorDestroy(Coordinator *coordinator) {
  coordinator->init = 0;
  coordinator->done = 0;
  coordinator->sendrank = -1;
  coordinator->recvrank = -1;

  LOG_MOD(NCCL_MOD, "CoordinatorDestroy");
  return ncclSuccess;
}

/**
 * SECTION 3: Communication Functions
 *  
 * coordinatorGetSendSize: Retrieves the size of the data to be sent for a specific channel. It checks that the sendtail 
 * (which tracks progress through the send buffer) is not ahead of the recvtail (tracking the received data) to ensure proper synchronization.
 * 
 * coordinatorSend & coordinatorRecv: Handle sending and receiving data for a specific channel. These functions update the sendtail and recvtail 
 * to reflect progress in data transmission and ensure that the communication task proceeds correctly. 
 * They also call update_done to check if the entire task has been completed.
*/

ncclResult_t coordinatorGetSendSize(Coordinator *coordinator, int cid,
                                       int &size) {
  auto &ch = coordinator->ranks[coordinator->sendrank].channels[cid];
  auto &chrecv = coordinator->ranks[coordinator->recvrank].channels[cid];
  if (ch.sendtail <= chrecv.recvtail) {
    size = ch.sendsizes[ch.sendtail];
  } else {
    size = -1;
    LOG_MOD(NCCL_MOD, "sendtail=%d > recvtail=%d", ch.sendtail, ch.recvtail);
  }
    LOG_MOD(NCCL_MOD, "CoordinatorGetSendSize: size=%d", size);
    return ncclSuccess;
}

ncclResult_t coordinatorSend(Coordinator *coordinator, int cid,
                                int size) {
  auto &ch = coordinator->ranks[coordinator->sendrank].channels[cid];
  if (ch.sendsizes[ch.sendtail] == size) {
    ch.sendtail++;
    update_done(coordinator);
  } else {
    LOG_MOD(NCCL_MOD, "send size unmatch actual: %d != expected: %d", size,
            ch.sendsizes[ch.sendtail]);
  }
  LOG_MOD(NCCL_MOD, "CoordinatorSend: size=%d, tail=%d", size, ch.sendtail);
  return ncclSuccess;
}

ncclResult_t coordinatorRecv(Coordinator *coordinator, int cid,
                                int size) {
  auto &ch = coordinator->ranks[coordinator->recvrank].channels[cid];
  if (ch.recvsizes[ch.recvtail] == size) {
    ch.recvtail++;
    update_done(coordinator);
  } else {
    LOG_MOD(NCCL_MOD, "recv size unmatch actual: %d != expected: %d", size,
            ch.recvsizes[ch.recvtail]);
  }
  LOG_MOD(NCCL_MOD, "CoordinatorRecv: size=%d, recvtail=%d", size,
          ch.recvtail);
  return ncclSuccess;
}

