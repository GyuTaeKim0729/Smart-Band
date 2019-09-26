#include "weights.h"
#include "arm_nnfunctions.h"
#include "arm_math.h"


#define NUM_MFCC_COEFFS 6
#define NUM_FRAMES 32

#define IN_DIM (NUM_FRAMES*NUM_MFCC_COEFFS)
#define OUT_DIM 4

#define CONV1_OUT_CH 32
#define CONV1_IN_X NUM_MFCC_COEFFS
#define CONV1_IN_Y NUM_FRAMES
#define CONV1_KX 6
#define CONV1_KY 6
#define CONV1_SX 1
#define CONV1_SY 2
#define CONV1_PX 2
#define CONV1_PY 2
#define CONV1_OUT_X 6
#define CONV1_OUT_Y 16
#define CONV1_BIAS_LSHIFT 2
#define CONV1_OUT_RSHIFT 11

#define CONV2_OUT_CH 16
#define CONV2_IN_X CONV1_OUT_X
#define CONV2_IN_Y CONV1_OUT_Y
#define CONV2_DS_KX 6
#define CONV2_DS_KY 6
#define CONV2_DS_SX 1
#define CONV2_DS_SY 1
#define CONV2_DS_PX 2
#define CONV2_DS_PY 2
#define CONV2_OUT_X CONV2_IN_X
#define CONV2_OUT_Y CONV2_IN_Y
#define CONV2_DS_BIAS_LSHIFT 1
#define CONV2_DS_OUT_RSHIFT 7
#define CONV2_PW_BIAS_LSHIFT 1
#define CONV2_PW_OUT_RSHIFT 7

#define CONV3_OUT_CH 16
#define CONV3_IN_X CONV2_OUT_X
#define CONV3_IN_Y CONV2_OUT_Y
#define CONV3_DS_KX 6
#define CONV3_DS_KY 6
#define CONV3_DS_SX 1
#define CONV3_DS_SY 1
#define CONV3_DS_PX 2
#define CONV3_DS_PY 2
#define CONV3_OUT_X CONV3_IN_X
#define CONV3_OUT_Y CONV3_IN_Y
#define CONV3_DS_BIAS_LSHIFT 2
#define CONV3_DS_OUT_RSHIFT 7
#define CONV3_PW_BIAS_LSHIFT 2
#define CONV3_PW_OUT_RSHIFT 7

#define CONV4_OUT_CH 16
#define CONV4_IN_X CONV3_OUT_X
#define CONV4_IN_Y CONV3_OUT_Y
#define CONV4_DS_KX 6
#define CONV4_DS_KY 6
#define CONV4_DS_SX 1
#define CONV4_DS_SY 1
#define CONV4_DS_PX 2
#define CONV4_DS_PY 2
#define CONV4_OUT_X CONV4_IN_X
#define CONV4_OUT_Y CONV4_IN_Y
#define CONV4_DS_BIAS_LSHIFT 2
#define CONV4_DS_OUT_RSHIFT 7
#define CONV4_PW_BIAS_LSHIFT 2
#define CONV4_PW_OUT_RSHIFT 7

#define CONV5_OUT_CH 16
#define CONV5_IN_X CONV4_OUT_X
#define CONV5_IN_Y CONV4_OUT_Y
#define CONV5_DS_KX 6
#define CONV5_DS_KY 6
#define CONV5_DS_SX 1
#define CONV5_DS_SY 1
#define CONV5_DS_PX 2
#define CONV5_DS_PY 2
#define CONV5_OUT_X CONV5_IN_X
#define CONV5_OUT_Y CONV5_IN_Y
#define CONV5_DS_BIAS_LSHIFT 3
#define CONV5_DS_OUT_RSHIFT 8
#define CONV5_PW_BIAS_LSHIFT 1
#define CONV5_PW_OUT_RSHIFT 6

#define FINAL_FC_BIAS_LSHIFT -1
#define FINAL_FC_OUT_RSHIFT 7

#define SCRATCH_BUFFER_SIZE (2*2*CONV1_OUT_CH*CONV2_DS_KX*CONV2_DS_KY + 2*CONV2_OUT_CH*CONV2_OUT_X*CONV2_OUT_Y)


void gesture_init(void);
void gesture_recognition(q7_t* out_data, q7_t* in_data);

