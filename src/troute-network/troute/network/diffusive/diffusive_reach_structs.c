#include <stdlib.h>
#include "../reach_structs.h"

void init_diffusive_reach(_DiffReach* reach, int num_segments)
{
  reach->reach.diff_reach.num_segments = num_segments;
  reach->reach.diff_reach._segments = (_Diff_Segment*) malloc(sizeof(_Diff_Segment)*(num_segments));
}

void free_diffusive_reach(_DiffReach* reach)
{
  if( reach != NULL && reach->reach.diff_reach._segments != NULL)
    free(reach->reach.diff_reach._segments);
}

void set_diffusive_segment(_DiffReach* reach, int index, long id,
    float dt, float dx, float bw, float tw, float twcc,
    float n, float ncc, float cs, float s0,
    float qdp, float velp, float depthp)
{
  if(index > -1 && index < reach->reach.diff_reach.num_segments)
  {
    _Diff_Segment segment;
    segment.id = id;
    segment.dt = dt;
    segment.dx = dx;
    segment.bw = bw;
    segment.tw = tw;
    segment.twcc = twcc;
    segment.n = n;
    segment.ncc = ncc;
    segment.cs = cs;
    segment.s0 = s0;
    segment.qdp = qdp;
    segment.velp = velp;
    segment.depthp = depthp;
    reach->reach.diff_reach._segments[index] = segment;
  }
  //FIXME else what?
}

_Diff_Segment get_diffusive_segment(_DiffReach* reach, int index)
{
  _Diff_Segment seg;
  if( reach != NULL && index > -1 && index < reach->reach.diff_reach.num_segments){
    if(reach->reach.diff_reach._segments != NULL )
      seg = reach->reach.diff_reach._segments[index];
  }
  return seg;
}
