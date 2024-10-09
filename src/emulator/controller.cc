#include "align.h"
#include "comm.h"
#include "emulator.h"
#include "helper.h"
#include "nccl.h"
#include <assert.h>
#include <cassert>
#include <cinttypes>
#include <map>
#include <math.h>
#include <sched.h>
#include <stdlib.h>
#include <string>

using namespace std;

Controller global_controller;


/**
 * This file represents the controller. The controller is responsible for the creation and 
 * tracking of tasks (Collective Communications). These tasks are given to the coordinator 
 * to coordinate the communication among ranks.
 * 
 * The controller also handles proxy communication
 */

/**
 * Section 1: Helper Methods
*/



static void calc_size_inkernel(int coll, int nelem, int tsize,vector<int> &res) {
  LOG_NOTE(NCCL_NOTE, "calc_size_inkernel: nelem=%d", nelem);
  int stepSize = (coll == ncclFuncAllGather || coll == ncclFuncAllReduce ||
                  coll == ncclFuncReduceScatter)
                     ? (131072 * 4 / tsize)
                     : 524288; // DEFAULT_BUFFSIZE(simple) / NCCL_STEP / tsize
  int SlicePerChunk = (coll == ncclFuncAllGather || coll == ncclFuncAllReduce ||
                       coll == ncclFuncReduceScatter)
                          ? 2
                          : 1;
  int StepPerSlice = (coll == ncclFuncAllGather || coll == ncclFuncAllReduce ||
                      coll == ncclFuncReduceScatter)
                         ? 2
                         : 1; //! i don't know why

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

static void calc_size_channel_AllReduce(int nranks, int ringindex, uint64_t count,
                              int nchannels, int mychannel, int nthreads,
                              int tsize, vector<int> &res) {
  //allreduce
  const int chunkSize = 524288;
  int bid = mychannel;
  int loopSize = nchannels * nranks * chunkSize;
  ssize_t size = count;
  int ringIx = ringindex;

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    ssize_t realChunkSize;
    // if proto == Simple
    realChunkSize =
        min((uint64_t)chunkSize, (uint64_t)DIVUP(size - gridOffset, nchannels * nranks));

    realChunkSize =
        ROUNDUP(realChunkSize, (nthreads - 32) * sizeof(uint64_t) / tsize);
    realChunkSize = int(realChunkSize);

    LOG_NOTE(NCCL_NOTE, "realChunkSize=%lu, nthreads=%d", realChunkSize,
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
    calc_size_inkernel(ncclFuncAllReduce, nelem, tsize,res);



    // k-2 steps: reduce and copy to next GPU
    for (int j = 2; j < nranks; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      offset = calcOffset(chunk);
      nelem = std::min(realChunkSize, size - offset);

      calc_size_inkernel(ncclFuncAllReduce, nelem, tsize, res);
    }

    // step k-1: reduce this buffer and data, which will produce the final
    // result that we store in this data and push to the next GPU
    chunk = ringIx + 0;
    offset = calcOffset(chunk);
    nelem = std::min(realChunkSize, size - offset);

    calc_size_inkernel(ncclFuncAllReduce, nelem, tsize, res);


    // k-2 steps: copy to next GPU
    for (int j = 1; j < nranks - 1; ++j) {
      chunk = modRanks(ringIx + nranks - j);
      offset = calcOffset(chunk);
      nelem = std::min(realChunkSize, size - offset);
      calc_size_inkernel(ncclFuncAllReduce, nelem, tsize, res);
    }
  }
  for (int i = 0; i < res.size(); i++) {
    res[i] *= tsize;
  }
}

static void calc_size_channel_AllGather(int nranks, int ringindex, uint64_t count,
                              int nchannels, int mychannel, int nthreads,
                              int tsize, vector<int> &res) {
  // allgather
  const int chunkSize =
      2097152;         // const ssize_t chunkSize =
                       // int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id ==
                       // NCCL_PROTO_SIMPLE ? ALLGATHER_CHUNKSTEPS : 1));
  int bid = mychannel; // const int bid = args->bid;
  int loopSize = nchannels * int(chunkSize);
  ssize_t size = count;

  int ringIx = ringindex;
  int _ringRanks[2];
  _ringRanks[0] = ringIx;
  _ringRanks[1] = ringIx ^ 1; // only can used when nranks ==2 !!!
  LOG_NOTE(NCCL_NOTE,
          "ringindex= %d, count=%lu,nchannels=%d, int mychannel=%d, int "
          "nthreads=%d,tsize=%d\n",
          ringindex, count, nchannels, mychannel, nthreads, tsize);

  LOG_NOTE(
      NCCL_NOTE,
      "nChennals: %d ; chunkSize: %d ; loopSize: %d ; size: %lu ; bid: %d\n",
      nchannels, chunkSize, loopSize, size, bid);

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    ssize_t realChunkSize;
    realChunkSize =
        min((uint64_t)chunkSize, (uint64_t)divUp(size - gridOffset, nchannels));
    LOG_NOTE(NCCL_NOTE, "realChunkSize-0: %lu\n", realChunkSize);
    realChunkSize =
        roundUp(realChunkSize, (nthreads - 32) * sizeof(uint64_t) /
                                   tsize); // warp_size==32; sizeof(t)=1
    realChunkSize = int(realChunkSize);
    LOG_NOTE(NCCL_NOTE, "realChunkSize=%lu, nthreads=%d", realChunkSize,
            nthreads);
    ssize_t chunkOffset = gridOffset + int(bid * realChunkSize);
    ssize_t offset;
    int nelem = min(realChunkSize, size - chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = _ringRanks[0];
    offset = chunkOffset + rankDest * size;

    // if (inputBuf + chunkOffset == outputBuf + offset) { // In place...?how to
    // get this info?
    //   prims.directSend(chunkOffset, offset, nelem);
    // } else {
    //   prims.directCopySend(chunkOffset, offset, nelem); // can like that? not
    //   same to allreduce.
    // }
    calc_size_inkernel(ncclFuncAllGather, nelem, tsize, res);
    // k-2 steps: copy to next GPU
    for (int j = 1; j < nranks - 1; ++j) {
      rankDest = _ringRanks[nranks - j];
      offset = chunkOffset + rankDest * size;

      calc_size_inkernel(ncclFuncAllGather, nelem, tsize,
                         res); // prims.directRecvCopySend(offset, nelem);
    }
    // Make final copy from buffer to dest.
    rankDest = _ringRanks[1];
    offset = chunkOffset + rankDest * size;
    // Final wait/copy.
    // prims.directRecv(offset, nelem); deleted
  }
  // allgather tsize=1
  //  for (int i = 0; i < res.size(); i++) {
  //    res[i] *= tsize;
  //  }
}

void calc_size_channel_Broadcast(int nranks, int ringindex, uint64_t count,
                                 int nchannels, int mychannel, int nthreads,
                                 int tsize, int root, vector<int> &res) {
  // broadcast

  if (root == -1)
    LOG_NOTE(NCCL_NOTE, "broadcast root=-1, something is wrong!!!");
  const int chunkSize =
      524288; // const ssize_t chunkSize =
              // int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id ==
              // NCCL_PROTO_SIMPLE ? BROADCAST_CHUNKSTEPS : 1));
  // BROADCAST_CHUNKSTEPS == 1

  int bid = mychannel; // const int bid = args->bid;
  ssize_t loopSize = nchannels * int(chunkSize);
  ssize_t size = count;

  int ringIx = ringindex;
  int rank = ringIx;         // const int rank = ring->userRanks[0];
  int nextRank = ringIx ^ 1; // const int nextRank = ring->userRanks[1];

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    ssize_t realChunkSize;
    // if (Proto::Id == NCCL_PROTO_SIMPLE)
    realChunkSize =
        min((uint64_t)chunkSize, (uint64_t)divUp(size - gridOffset, nchannels));
    realChunkSize =
        roundUp(realChunkSize, (nthreads - 32) * sizeof(uint64_t) / tsize);
    realChunkSize = int(realChunkSize);

    ssize_t offset = gridOffset + int(bid * realChunkSize);
    int nelem = min(realChunkSize, size - offset);

    if (rank == root) {
      // if (tid == 0) {
      //   printf("[nccl] Broadcast @root offset %d nelem %ld\n", root,
      //   offset,
      //          nelem);
      // }
      calc_size_inkernel(ncclFuncBroadcast, nelem, tsize, res);
      // if (inputBuf == outputBuf) {
      //   prims.send(offset, nelem);
      // } else {
      //   prims.copySend(offset, offset, nelem);
      // }
    } else if (nextRank == root) {
      // if (tid == 0) {
      //   printf("[nccl] Broadcast @next_to_root offset %d nelem %ld\n",
      //   root,
      //          offset, nelem);
      // }
      // prims.recv(offset, nelem);
    } else {
      calc_size_inkernel(ncclFuncBroadcast, nelem, tsize, res);
      // prims.recvCopySend(offset, nelem);
    }
  }
}

void calc_size_channel_ReduceScatter(int nranks, int ringindex, uint64_t count,
                                 int nchannels, int mychannel, int nthreads,
                                 int tsize, vector<int> &res) {

  const int chunkSize =
      524288; // int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? REDUCESCATTER_CHUNKSTEPS : 1));

  int bid = mychannel; // const int bid = args->bid;
  ssize_t loopSize = nchannels * int(chunkSize);
  ssize_t size = count;

  int ringIx = ringindex;
  int rank = ringIx;         // const int rank = ring->userRanks[0];
  int nextRank = ringIx ^ 1; // const int nextRank = ring->userRanks[1];

  int _ringRanks[2];
  _ringRanks[0] = ringIx;
  _ringRanks[1] = ringIx ^ 1; // only can used when nranks ==2 !!!

  for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
    ssize_t realChunkSize;
    //if (Proto::Id == NCCL_PROTO_SIMPLE)
    realChunkSize = min((uint64_t)chunkSize, (uint64_t)divUp(size-gridOffset, nchannels));
    realChunkSize = roundUp(realChunkSize, (nthreads-32)*sizeof(uint64_t)/tsize);
    realChunkSize = int(realChunkSize);

    ssize_t chunkOffset = gridOffset + bid*int(realChunkSize);

    /////////////// begin ReduceScatter steps ///////////////
    ssize_t offset;
    int nelem = min(realChunkSize, size-chunkOffset);
    int rankDest;

    // step 0: push data to next GPU
    rankDest = _ringRanks[nranks-1];
    offset = chunkOffset + rankDest * size;
    //prims.send(offset, nelem);
    calc_size_inkernel(ncclFuncReduceScatter, nelem, tsize, res);

    // k-2 steps: reduce and copy to next GPU
    for (int j=2; j<nranks; ++j) {
      rankDest = _ringRanks[nranks-j];
      offset = chunkOffset + rankDest * size;
      //prims.recvReduceSend(offset, nelem);
      calc_size_inkernel(ncclFuncReduceScatter, nelem, tsize, res);

    }

    // step k-1: reduce this buffer and data, which will produce the final result
    rankDest = _ringRanks[0];
    offset = chunkOffset + rankDest * size;
    //prims.recvReduceCopy(offset, chunkOffset, nelem, /*postOp=*/true);
  }
  for (int i = 0; i < res.size(); i++) {
    res[i] *= tsize;
  }
}

static inline __attribute__((always_inline)) void
calc_sendsize_channel(int nranks, int myrank, uint64_t count, int nchannels,
                      int mychannel, int nthreads, int coll, vector<int> &res,
                      int root) {
  // root is only for broadcast
  //   auto &ringmap = global_topology.ringmap;
  //   assert(ringmap.count(make_pair(myrank, mychannel)) > 0);
  // int myringix = ringmap[make_pair(myrank, mychannel)];
  //! todo multiple gpu per proc
  int myringix = myrank;
  if (coll == ncclFuncAllReduce)
    calc_size_channel_AllReduce(nranks, myringix, count, nchannels, mychannel,
                                nthreads, sizeof(float), res);
  else if (coll == ncclFuncReduceScatter)
    calc_size_channel_ReduceScatter(nranks, myringix, count, nchannels,
                                    mychannel, nthreads, sizeof(float), res);
  else if (coll == ncclFuncAllGather)
    calc_size_channel_AllGather(nranks, myringix, count, nchannels, mychannel,
                                nthreads, 1, res); // allgather
  else if (coll == ncclFuncBroadcast)
    calc_size_channel_Broadcast(nranks, myringix, count, nchannels, mychannel,
                                nthreads, 1, root, res); // broadcast
  else {
    printf("unsupported coll type!");
    abort();
  }
  std::string szs;
  for (int i = 0; i < res.size(); ++i) {
    szs = szs + " " + std::to_string(res[i]);
  }
  LOG_NOTE(NCCL_NOTE, "Calculated send sizes for ringix:%d rank %d: %s, ch=%d",
          myringix, myrank, szs.c_str(), mychannel);
}

static inline __attribute__((always_inline)) void
calc_recvsize_channel(int nranks, int myrank, uint64_t count, int nchannels,
                      int mychannel, int nthreads, int coll, vector<int> &res,
                      int root) {
  //   auto &ringmap = global_topology.ringmap;
  //   assert(ringmap.count(make_pair(myrank, mychannel)) > 0);
  //   int target = global_topology.prev[myrank];
  //   int target_ringix = ringmap[make_pair(target, mychannel)];
  //! todo
  int target = (myrank + 1) % nranks;
  int target_ringix = target;

  if (coll == ncclFuncAllReduce)
    calc_size_channel_AllReduce(nranks, target_ringix, count, nchannels,
                                mychannel, nthreads, sizeof(float), res);
  else if (coll == ncclFuncReduceScatter)
    calc_size_channel_ReduceScatter(nranks, target_ringix, count, nchannels,
                                    mychannel, nthreads, sizeof(float), res);
  else if (coll == ncclFuncAllGather)
    calc_size_channel_AllGather(nranks, target_ringix, count, nchannels,
                                mychannel, nthreads, 1, res); // allgather
  else  if (coll == ncclFuncBroadcast)
    calc_size_channel_Broadcast(nranks, target_ringix, count, nchannels,
                                mychannel, nthreads, 1, root, res); // broadcast
  else {
    printf("unsupported coll type!");
    abort();
  }
  
  std::string szs;
  for (int i = 0; i < res.size(); ++i) {
    szs = szs + " " + std::to_string(res[i]);
  }
  LOG_NOTE(NCCL_NOTE,
          "Calculated recv sizes for ringix:%d targetrk:%d, rank %d: %s, ch=%d",
          target_ringix, target, myrank, szs.c_str(), mychannel);

static inline __attribute__((always_inline)) void
channelInit(ChannelInfo *ch, RankInfo *rankinfo, int nranks, int myrank,
            int chid, uint64_t count, int nchannels, int nthreads, int coll,
            int root) {
  ch->bid = chid;
  ch->sendsizes = vector<int>();
  ch->recvsizes = vector<int>();
  ch->send = rankinfo->send;
  ch->recv = rankinfo->recv;
  if (rankinfo->send) {
    calc_sendsize_channel(nranks, myrank, count, nchannels, chid, nthreads,
                          coll, ch->sendsizes, root);
  }
  if (rankinfo->recv) {
    calc_recvsize_channel(nranks, myrank, count, nchannels, chid, nthreads,
                          coll, ch->recvsizes, root);
  }
  ch->sendtail = 0;
  ch->recvtail = 0;
  ch->senddone = 0;
  ch->recvdone = 0;
  LOG_NOTE(NCCL_NOTE,
          "channelInit: myrank=%d, chid=%d, send=%d, recv=%d, root=%d\n",
          rankinfo->myrank, chid, ch->send, ch->recv, root);
}

static inline __attribute__((always_inline)) void
rankInit(RankInfo *rankinfo, EmulatorTask *task, CommInfo *comm,
         int rank) {
  int nchannels = task->info.nchannels;
  int nthreads = task->info.nthreads;
  int nranks = comm->nranks;
  uint64_t count = task->info.count;
  rankinfo->done = 0;
  rankinfo->myrank = rank;
  rankinfo->send = 0;
  rankinfo->recv = 0;
  if (rankinfo->myrank == task->sendrank) {
    rankinfo->send = 1;
  }
  if (rankinfo->myrank == task->recvrank) {
    rankinfo->recv = 1;
  }
  //! todo consider multiple rank per proc
  assert(rankinfo->send == 1 && rankinfo->recv == 1);
  LOG_NOTE(NCCL_NOTE, "rankInit: myrank=%d, send=%d, recv=%d", rankinfo->myrank,
          rankinfo->send, rankinfo->recv);
  rankinfo->channels = vector<ChannelInfo>();
  for (int i = 0; i < nchannels; ++i) {
    ChannelInfo ch;
    channelInit(&ch, rankinfo, nranks, rank, i, count, nchannels, nthreads,
                task->info.coll,task->info.root);
    //! todo fix tsize
    rankinfo->channels.push_back(ch);
  }
}

static inline __attribute__((always_inline)) int
bypassCheckInternal(TaskInfo info, uint64_t unique_id) {
  return KERNEL_BYPASS == 1 &&
         (info.coll == ncclFuncAllReduce || info.coll == ncclFuncAllGather ||
          info.coll == ncclFuncBroadcast ||
          info.coll == ncclFuncReduceScatter) &&
         unique_id >= 0;
}

static inline __attribute__((always_inline)) int
emulatorTaskInit(EmulatorTask *task, CommInfo *comm, ncclInfo *info) {
  Info2Task(info, &task->info);
  task->init = 1;
  task->done = 0;
  task->ranks = map<int, RankInfo>();
  task->sendrank = comm->mynode;
  task->recvrank = comm->mynode;
  if (bypassCheckInternal(task->info, task->info.unique_id)) {
    //! fix me here we assume 2 node, 1 rank per node
    for (int i = 0; i < comm->nrankpernode; ++i) {
      int rank = comm->nrankpernode * comm->mynode + i;
      RankInfo rankinfo;
      rankInit(&rankinfo, task, comm, rank);
      task->ranks[rank] = rankinfo;
      LOG_NOTE(NCCL_NOTE, "emulatorTaskInit: rank=%d", rank);
    }
    //! todo sendrecv init
  } else {
    LOG_NOTE(NCCL_NOTE, "emulatorTaskInit: bypassed task, unique_id=%lu",
            task->info.unique_id);
  }
  LOG_NOTE(NCCL_NOTE, "emulatorTaskInit: unique_id=%lu", task->info.unique_id);
  return 0;
}

static inline __attribute__((always_inline)) int
emulatorTaskDestroy(EmulatorTask *task) {
  task->init = 0;
  task->done = 0;
  task->ranks.clear();
  task->sendrank = -1;
  task->recvrank = -1;
  return 0;
}

static inline __attribute__((always_inline)) int
check_done_ch(ChannelInfo *ch) {
  return ch->sendtail == ch->sendsizes.size() && ch->senddone;
  // return ch->sendtail == ch->sendsizes.size() &&
  //        ch->recvtail == ch->recvsizes.size() && ch->senddone && ch->recvdone;
}

static inline __attribute__((always_inline)) int
check_done_rank(RankInfo *rank) {
  if (rank->done) {
    return 1;
  }
  int done = 1;
  for (int i = 0; i < rank->channels.size(); ++i) {
    done = done & check_done_ch(&rank->channels[i]);
  }
  rank->done = done;
  LOG_NOTE(NCCL_NOTE, "check_done_rank: done=%d, rank=%d", done, rank->myrank);
  return done;
}

static inline __attribute__((always_inline)) int
check_done_task(EmulatorTask *task) {
  if (task->done) {
    return 1;
  }
  int done = 1;
    LOG_NOTE(NCCL_NOTE, "start check_done_task: done=%d, unique_id=%lu, rksize=%lu",
            done, task->info.unique_id, task->ranks.size());
  for (int i = 0; i < task->ranks.size(); ++i) {
    done = done & check_done_rank(&task->ranks[i]);
  }
  if (done) {
    task->done = 1;
    // printf("[sync] task unique_id = %lu done\n", task->info.unique_id);
    LOG_NOTE(NCCL_NOTE, "check_done_task: done=%d, unique_id=%lu, rksize=%lu",
            done, task->info.unique_id, task->ranks.size());
  }
  return done;
}

static inline __attribute__((always_inline)) int
syncTask(EmulatorTask *task) {
  if (task->done == 1) {
    return 1;
  } else {
    check_done_task(task);
    return task->done;
  }
}

ncclResult_t ncclModStreamSyncFunc(Controller *controller, cudaStream_t s) {
  if (KERNEL_BYPASS != 1) {
    LOG_NOTE(NCCL_NOTE, "ncclModStreamSyncFunc: bypass is off, return");
    return ncclSuccess;
  }
  emulator_lock.lock();
  if (controller->stream2ids.count(s) == 0) {
    LOG_NOTE(NCCL_LOG_WARN, "ncclModStreamSyncFunc: stream not found");
    return ncclSuccess;
  }
  static int last = 0;
  auto ids = controller->stream2ids[s];


  vector<uint64_t> rest_ids;
  for (int i = last; i < ids.size(); ++i) {
    rest_ids.push_back(ids[i]);
  }
  int flag = 1;
  while (1) {
    flag = 1;
    for (auto i : rest_ids) {
      assert(controller->id2task.count(i) > 0);
      auto &task = controller->id2task[i];
      if (task.done || task.info.bypass == 0) {
        continue;
      }
      flag = flag & syncTask(&task);
    }
    if (flag) {
      break;
    } else {
      emulator_lock.unlock();
      sched_yield();
      emulator_lock.lock();
    }
  }
  last = ids.size();
  emulator_lock.unlock();
  // controller->id2task.clear();
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclModStreamSync, cudaStream_t s);
ncclResult_t ncclModStreamSync(cudaStream_t s) {
  return ncclModStreamSyncFunc(&global_controller, s);
}


static uint64_t gen_unique_id() {
  static uint64_t unique_id = 0;
  return ++unique_id;
}

static int gen_stream_id() {
  static int stream_id = 0;
  return ++stream_id;
}

ncclResult_t addTask(Controller *controller, ncclInfo *info) {
  info->unique_id = gen_unique_id();
  controller->stream2ids[info->stream].push_back(info->unique_id);
  if (controller->stream2int.count(info->stream) == 0) {
    controller->stream2int[info->stream] = gen_stream_id();
  }
  LOG_NOTE(NCCL_NOTE, "addTask for unique_id: %lu in stream %lu",
          info->unique_id, (uint64_t)info->stream);
  return ncclSuccess;
}

ncclResult_t initTask(Controller *controller, ncclInfo *info) {
  EmulatorTask task;
  auto unique_id = info->unique_id;
  LOG_NOTE(NCCL_NOTE, "initTask for unique_id: %lu", unique_id);
  assert(controller->id2task.count(unique_id) == 0);
  emulatorTaskInit(&task, controller->comm, info);
  controller->id2task[task.info.unique_id] = task;
  // printf("[emulator] task unique_id = %lu inited\n", task.info.unique_id);
  int nchannels = task.info.nchannels;
  for (int i = 0; i < nchannels; ++i) {
    if (controller->cid2bypassed.count(i) == 0) {
      controller->cid2bypassed[i] = make_pair(0, 0);
    }
  }
  return ncclSuccess;
}

ncclResult_t queryTask(Controller *controller, uint64_t unique_id,
                          TaskInfo *task) {
  LOG_NOTE(NCCL_NOTE, "queryTask for unique_id: %lu", unique_id);
  auto it = controller->id2task.find(unique_id);
  if (it != controller->id2task.end()) {
    *task = it->second.info;
    return ncclSuccess;
  } else {
    LOG_NOTE(NCCL_LOG_WARN, "queryTask: task not found");
    return ncclSuccess;
  }
}

ncclResult_t removeTask(Controller *controller, uint64_t unique_id) {
  LOG_NOTE(NCCL_NOTE, "removeTask for unique_id: %lu", unique_id);
  if (controller->id2task.count(unique_id) > 0) {
    controller->id2task.erase(unique_id);
    return ncclSuccess;
  } else {
    LOG_NOTE(NCCL_LOG_WARN, "removeTask: task not found");
    abort();
    return ncclSuccess;
  }
}

ncclResult_t bypassCheck(Controller *controller, uint64_t unique_id,
                            int &bypass, std::string msg) {
  if (controller->id2task.count(unique_id) <= 0) {
    fprintf(stderr,
            "ERROR: task for unique_id: %lu not found from %s, size = %lu\n",
            unique_id, msg.c_str(), controller->id2task.size());
  }
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  bypass = bypassCheckInternal(task.info, unique_id);
  task.info.bypass = bypass;
  //! fix me
  LOG_NOTE(NCCL_NOTE, "bypassCheck for unique_id: %lu, bypass = %d", unique_id,
          bypass);
  return ncclSuccess;
}

ncclResult_t globalInit(Controller *controller, ncclComm *comm) {

  controller->comm = new CommInfo();
  controller->comm->nranks = comm->nRanks;
  controller->comm->mynode = MY_NODE;
  controller->comm->nnodes = NUM_NODES;
  controller->comm->nrankpernode = comm->nRanks / NUM_NODES;
  controller->id2task = map<uint64_t, EmulatorTask>();
  printf("[emulator] global init done");

  controller->stream2ids = map<cudaStream_t, vector<uint64_t>>();

  controller->coordinator = &global_coordinator;

  controller->topology = &global_topology;
  //! todo init topology and coordinator here!

  return ncclSuccess;
}

//proxy functions
__attribute__((always_inline)) int
proxyGetSendSize(Controller *controller, int unique_id, int cid,
                    int &size) {
  LOG_NOTE(NCCL_NOTE, "proxyGetSendSize for unique_id: %d, cid: %d, size: %d",
          unique_id, cid, size);
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  auto &rank = task.ranks[task.sendrank];
  [[maybe_unused]] auto &recvch = task.ranks[task.recvrank].channels[cid];
  auto &ch = rank.channels[cid];
  // if (ch.sendtail <= recvch.recvtail) {
  size = ch.sendsizes[ch.sendtail];
  // } else {
  //   size = -1;
  // }
  return 0;
 }

__attribute__((always_inline)) int
proxySend(Controller *controller, int unique_id, int cid, int size) {
   LOG_NOTE(NCCL_NOTE, "proxySend for unique_id: %d, cid: %d, size: %d",
           unique_id, cid, size);
   assert(controller->id2task.count(unique_id) > 0);
   auto &task = controller->id2task[unique_id];
   auto &rank = task.ranks[task.sendrank];
   auto &ch = rank.channels[cid];
   LOG_NOTE(NCCL_NOTE, "proxySend for ch.recvsizes[%d]: %d\n",ch.sendtail,ch.sendsizes[ch.sendtail]);
   assert(ch.sendsizes[ch.sendtail] == size);
   ch.sendtail++;

   return 0;
}

__attribute__((always_inline)) int
proxyRecv(Controller *controller, int unique_id, int cid, int size) {
  LOG_NOTE(NCCL_NOTE, "proxyRecv for unique_id: %d, cid: %d, size: %d",
          unique_id, cid, size);
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  auto &rank = task.ranks[task.recvrank];
  auto &ch = rank.channels[cid];
  LOG_NOTE(NCCL_NOTE, "proxyRecv for ch.recvsizes[%d]: %d\n",ch.recvtail,ch.recvsizes[ch.recvtail]);
  assert(ch.recvsizes[ch.recvtail] == size);
  ch.recvtail++;
  return 0;
}

__attribute__((always_inline)) int proxySendDone(Controller *controller,
                                                    int unique_id, int cid,
                                                    uint64_t bypassed) {
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  auto &rank = task.ranks[task.sendrank];
  auto &ch = rank.channels[cid];

  assert(ch.sendtail == ch.sendsizes.size() && ch.senddone == 0);
  ch.senddone = 1;
  auto &c = controller->cid2bypassed[cid];
  c.first += bypassed;
  LOG_NOTE(NCCL_NOTE, "proxySendDone for unique_id: %d, cid: %d, inc = %lu",
          unique_id, cid, bypassed);
  return 0;
}

__attribute__((always_inline)) int proxyRecvDone(Controller *controller,
                                                    int unique_id, int cid,
                                                    uint64_t bypassed) {
  assert(controller->id2task.count(unique_id) > 0);
  auto &task = controller->id2task[unique_id];
  auto &rank = task.ranks[task.recvrank];
  auto &ch = rank.channels[cid];
  LOG_NOTE(NCCL_NOTE, "proxyRecvDone  start ch.recvtail: %d, ch.recvsizes.size(): %d  ch.recvdone: %d\n",ch.recvtail, ch.recvsizes.size(),(int)ch.recvdone);  
  assert(ch.recvtail == ch.recvsizes.size() && ch.recvdone == 0);
  ch.recvdone = 1;
  auto &c = controller->cid2bypassed[cid];
  c.second += bypassed;
  LOG_NOTE(NCCL_NOTE, "proxyRecvDone for unique_id: %d, cid: %d, inc = %lu",
          unique_id, cid, bypassed);
  return 0;
}

__attribute__((always_inline)) int
proxyBypassedSend(Controller *controller, int unique_id, int cid,
                     uint64_t &bypassed) {
  auto &c = controller->cid2bypassed[cid];
  bypassed = c.first;
  return 0;
}

__attribute__((always_inline)) int
proxyBypassedRecv(Controller *controller, int unique_id, int cid,
                     uint64_t &bypassed) {
  auto &c = controller->cid2bypassed[cid];
  bypassed = c.second;
  return 0;
}