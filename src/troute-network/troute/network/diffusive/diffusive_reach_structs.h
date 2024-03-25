#ifndef DIFF_REACH_STRUCTS_H
#define DIFF_REACH_STRUCTS_H
/*
    C Structures
*/
#include "../reach_structs.h"

typedef struct _Diff_Segment{
  long id;
  float dt, dx, bw, tw, twcc, n, ncc, cs, s0;
  float qdp, velp, depthp;
} _Diff_Segment;

typedef struct _Diff_Reach{
  _Diff_Segment* _segments;
  int num_segments;
} _Diff_Reach;

#endif //DIFF_REACH_STRUCTS_H
