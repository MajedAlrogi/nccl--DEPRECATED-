#ifndef EMULATOR_H
#define EMULATOR_H

#include "driver_types.h"
#include "nccl.h"
#include "proxy.h"
#include <cstdint>
#include <map>
#include <mutex>
#include <set>
#include <sys/types.h>
#include <vector>

/**
    The controller initiates and tracks tasks, while the coordinator manages the detailed communication workflow for each task.
**/



// forward declarations
struct Coordinator;
struct EmuTopology;
struct Controller;

// begin global variables 
// environment variables: Specified in the config script
extern int KERNEL_BYPASS;
extern int NUM_NODES;
extern int MY_NODE;

//Global variables. [descriptions of their roles below]
extern Coordinator global_coordinator;
extern EmuTopology global_topology;
extern Controller global_controller;

//this lock manages synchronization of the emulator
extern std::mutex emulator_lock;

ncclResult_t getAllEnvVars(); //returns a struct that contains the environment variables
// end global variables


//***Coordinator related structs***

// channel represents a connection between two ranks
struct ChannelInfo {
  int bid;
  int send;
  int recv;
  std::vector<int> sendsizes;
  std::vector<int> recvsizes;
  int sendtail;
  int recvtail;
  int senddone;
  int recvdone;
};

// rank represents a gpu device
struct RankInfo {
  int myrank;
  int send;
  int recv;
  int done;
  std::vector<ChannelInfo> channels;
};

// This struct contains information about the configuration of the cluster
struct CommInfo {
  int nranks;
  int nnodes;
  int nrankpernode;
  int mynode;
};

// A Task reoresents a collective communication
struct TaskInfo {
  uint64_t count;// number of elements
  int tsize;     // size of each element
  int coll;      // i.e. allreduce
  int reduceOp;  // i.e. sum
  int algo;      // i.e. ring
  int proto;     // i.e. Simple
  int nchannels;
  int nthreads;
  uint64_t unique_id; //identifier
  int bypass;
  int root;
};
//This struct contains the information needed to finish a Task
struct EmulatorTask {
  int done;
  int init;
  int sendrank;
  int recvrank;
  TaskInfo info;
  std::map<int, RankInfo> ranks;
};
//The global coordinator is primarily responsible for managing the state of communication for each collective operation.
struct Coordinator {
  int done;
  int init;
  int sendrank;
  int recvrank;
  CommInfo comm;
  TaskInfo task;
  std::map<int, RankInfo> ranks;

  // <channelId, ringIndex>
  std::map<int, int> sendRingMap;
  std::map<int, int> recvRingMap;
  std::map<uint64_t, EmulatorTask> id2task;

  ncclProxyOp *proxyOp;
  ncclInfo *info;
};

//***Coordinator functions***

ncclResult_t coordinatorInit(Coordinator *coordinator,
                                ncclProxyOp *proxyOp, ncclInfo *info);

ncclResult_t coordinatorDestroy(Coordinator *coordinator);

ncclResult_t coordinatorGetSendSize(Coordinator *coordinator, int cid,
                                       int &size);

ncclResult_t coordinatorSend(Coordinator *coordinator, int cid, int size);

ncclResult_t coordinatorRecv(Coordinator *coordinator, int cid, int size);

// end coordinator

typedef enum {
  UNINITED = 0,
  META_INITED = 1,
  PER_CALL_INITED = 2,
} topoInitState;


struct EmuTopology {
  topoInitState init;
  int nranks;
  int nnodes;
  int nrankpernode;
  int nchannels;

  // ranks in this node
  std::set<int> myranks;
  std::map<int, int> prev;
  std::map<int, int> next;
  // <rank, channel> -> <ringIndex>
  std::map<std::pair<int, int>, int> ringmap;
};

ncclResult_t emuTopologyInit(EmuTopology *topology, ncclProxyOp *proxyOp,
                             ncclInfo *info);

ncclResult_t emuTopologyUpdateMap(EmuTopology *topology, int rank, int channel,
                                  ncclRing *ring, int *ringranks, int nranks);

ncclResult_t emuTopologyDestroy(EmuTopology *topology);


// begin controller

struct Controller {
  std::map<uint64_t, EmulatorTask> id2task;
  std::map<cudaStream_t, std::vector<uint64_t>> stream2ids;
  std::map<cudaStream_t, int> stream2int;
  std::map<int, std::pair<uint64_t, uint64_t>>
      cid2bypassed; // cid: <send, recv>
  Coordinator *coordinator;
  EmuTopology *topology;
  CommInfo *comm;
  uint64_t bypassed_send;
  uint64_t bypassed_recv;
};

ncclResult_t AddTask(Controller *controller, ncclInfo *info);

ncclResult_t InitTask(Controller *controller, ncclInfo *info);

ncclResult_t QueryTask(Controller *controller, uint64_t unique_id,
                          TaskInfo *task);

ncclResult_t RemoveTask(Controller *controller, uint64_t unique_id);

ncclResult_t BypassCheck(Controller *controller, uint64_t unique_id,
                            int &bypass, std::string msg);

ncclResult_t GlobalInit(Controller *controller, ncclComm *comm);
// end controller

// begin proxy

int ProxyGetSendSize(Controller *controller, int unique_id, int cid,
                        int &size);

int ProxySend(Controller *controller, int unique_id, int cid, int size);

int ProxyRecv(Controller *controller, int unique_id, int cid, int size);

int ProxySendDone(Controller *controller, int unique_id, int cid,
                     uint64_t bypassed);

int ProxyRecvDone(Controller *controller, int unique_id, int cid,
                     uint64_t bypassed);

int ProxyBypassedSend(Controller *controller, int unique_id, int cid,
                         uint64_t &bypassed);

int ProxyBypassedRecv(Controller *controller, int unique_id, int cid,
                         uint64_t &bypassed);

#endif // EMULATOR_H
