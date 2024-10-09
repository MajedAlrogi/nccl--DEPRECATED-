#include "helper.h"

using namespace std;

void Info2Task(ncclInfo *info, TaskInfo *task) {
  // Basic field assignments
  task->count = info->count;
  task->root = info->root;
  task->coll = info->coll;
  task->reduceOp = info->op;
  task->algo = info->algorithm;
  task->proto = info->protocol;
  task->nchannels = info->nChannels;
  task->nthreads = info->nThreads;
  task->unique_id = info->unique_id;

  // Determine tsize based on datatype using switch-case for better organization
  switch (info->datatype) {
    case ncclInt8:
    case ncclUint8:
      task->tsize = 1;
      break;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      task->tsize = 4;
      break;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      task->tsize = 8;
      break;
    case ncclFloat16:
      task->tsize = 2;
      break;
    default:
      task->tsize = 4; // Default value for unknown types
  }

  // Logging the task information for debugging or monitoring purposes
  LOG_NOTE(NCCL_NOTE,
          "Info2Task: count %lu, tsize %d, coll %d, reduceOp %d, algo %d, proto "
          "%d, nchannels %d, nthreads %d, unique_id %lu, root %d",
          task->count, task->tsize, task->coll, task->reduceOp, task->algo,
          task->proto, task->nchannels, task->nthreads, task->unique_id, task->root);
}