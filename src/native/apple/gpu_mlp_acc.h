/* gpu_mlp_acc.h — the fused MLP epochs' ping-ponged loss and correct-count
 * accumulators, shared by gpu_mlp.m (which defines them, beside the
 * accumulators themselves) and gpu_mlp3.m. See mlpEpochAccPingPong. */

#ifndef JAI_GPU_MLP_ACC_H
#define JAI_GPU_MLP_ACC_H

#ifdef __APPLE__

#include "native/apple/gpu_internal.h"

bool mlpEpochAccPingPong(void);
void mlpWireEpochAccs(NSMutableDictionary *feedsA, NSMutableDictionary *feedsB,
                      NSMutableDictionary *resultsA, NSMutableDictionary *resultsB,
                      bool firstIsA, MPSGraphTensor *accIn, MPSGraphTensor *corrIn,
                      MPSGraphTensor *accOut, MPSGraphTensor *corrOut,
                      MPSGraphTensorData *liveAcc, MPSGraphTensorData *liveCorr,
                      MPSGraphTensorData *scratchAcc, MPSGraphTensorData *scratchCorr);
void mlpNoteEpochAccsLocked(JaiGpuBuffer *lossAcc, JaiGpuBuffer *correctAcc);
bool mlpSettleEpochAccsLocked(uint32_t steps, JaiGpuBuffer *lossAcc, size_t lossOff,
                              JaiGpuBuffer *correctAcc, size_t correctOff);

#endif /* __APPLE__ */
#endif /* JAI_GPU_MLP_ACC_H */
