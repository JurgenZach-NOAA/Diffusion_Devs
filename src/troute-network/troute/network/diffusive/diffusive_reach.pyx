cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from troute.network.reach cimport compute_type, _Reach

cdef extern from "diffusive_reach_structs.c":
  void init_diffusive_reach(_Reach* reach, int num_segments)
  void free_diffusive_reach(_Reach* reach)
  void set_diffusive_segment(_Reach* reach, int index, long id,
      float dt, float dx, float bw, float tw, float twcc,
      float n, float ncc, float cs, float s0,
      float qdp, float velp, float depthp)
  _Diff_Segment get_diffusive_segment(_Reach* reach, int index) nogil

cdef class Diff_Segment(Segment):
  """
    A single segment
  """

  def __init__(self, id, dt, dx, bw, tw, twcc, n, ncc, cs, s0, qdp, velp, depthp):
    """
      TODO: implement topology of Diffusive segments
    """
    super().__init__(id, -1, -1)
    self.dt = dt
    self.dx = dx
    self.bw = bw
    self.tw = tw
    self.twcc = twcc
    self.n = n
    self.ncc = ncc
    self.cs = cs
    self.s0 = s0
    self.qdp = qdp
    self.velp = velp
    self.depthp = depthp

cdef class Diff_Reach(Reach):
  """
    Equivalent to MC_Reach
  """

  def __init__(self, segments, long[::1] upstream_ids):
    """
      segments: ORDERED list of segments that make up the reach
    """
    #for now mc reaches aren't explicity identified, their segments are pass -1 for id
    super().__init__(-1, upstream_ids, compute_type.MC_REACH)

    self._num_segments = len(segments)
    init_diffusive_reach(&self._reach, self._num_segments)
    for i, s in enumerate(segments):
      #TODO  what about segment id???
      set_diffusive_segment(&self._reach, i, s.id,
                     s.dt, s.dx, s.bw, s.tw,
                     s.twcc, s.n, s.ncc, s.cs,
                     s.s0, s.qdp, s.velp, s.depthp)

    #self._segments = segments
    self.num_segments = self._num_segments

  #TODO implement __cinit__ for init_mc_reach?
  def __dealloc__(self):
    """

    """
    free_diffusive_reach(&self._reach)

  cdef route(self):
    """
      TODO implement?
    """
    #do a few things with gil
    #with nogil:
    #  pass
    pass

  def __getitem__(self, index):
    #TODO implement slicing, better errors
    if(index < 0):
      index = index + self._num_segments
    if(index > -1 and index <self._num_segments):
      return get_diffusive_segment(&self._reach, index)
    else:
      raise(IndexError)

  def __len__(self):
    return self._num_segments
