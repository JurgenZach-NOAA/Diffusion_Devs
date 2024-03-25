# cython: language_level=3, boundscheck=True, wraparound=False, profile=True

import numpy as np
from itertools import chain
from operator import itemgetter
from array import array
from numpy cimport ndarray
from libc.math cimport isnan, NAN
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf

cimport troute.routing.fast_reach.reach as reach

@cython.boundscheck(False)
# Equivalent of compute_reach_kernel in MC case
cdef void compute_diff_kernel(const float[:] timestep_ar_g, const int[:] mem_size_buf, const int[:] usgs_da_reach_g, const int[:,:] frnw_ar_g, const int[:,:] size_bathy_g, const float[:] dbcd_g, const float[:] para_ar_g, const float[:,:,:] diff_buff, float[:,:] so_ar_g, const float[:,:] ubcd_g, const float[:,:] qtrib_g, const float[:,:] usgs_da_g, const float[:,:,:] qlat_g, const float[:,:,:,:] bathy_buf, const float[:,:] crosswalk_g, float[:,:,:,:] output_buf) nogil:


    '''
    
    Passed variables

    timestep_ar_g: Array (float) of times and timesteps, as follows:
                        dtini           = timestep_ar_g(1) ! initial timestep duration [sec]
                        t0              = timestep_ar_g(2) ! simulation start time [hr]
                        tfin            = timestep_ar_g(3) ! simulation end time [hr]
                        saveInterval    = timestep_ar_g(4) ! output recording interval [sec]
                        dt_ql           = timestep_ar_g(5) ! lateral inflow data time step [sec]
                        dt_ub           = timestep_ar_g(6) ! upstream boundary time step [sec]
                        dt_db           = timestep_ar_g(7) ! downstream boundary time step [sec]
                        dt_qtrib        = timestep_ar_g(8) ! tributary data time step [sec]
                        dt_da           = timestep_ar_g(9) ! data assimilation data time step [sec]

    mem_size_buf: 1D integer array of array sizes:
    
                        mxncomp_g     : maximum number of nodes in a single reach
                        nrch_g        : number of reaches in the network
                        nts_ql_g      : number of qlateral timesteps
                        nts_ub_g      : upstream boundary time steps
                        nts_db_g      : number of timesteps in downstream boundary data
                        nts_qtrib_g   : number of timesteps in tributary time series data
                        nts_da_g      : number of DA time steps
                        ntss_ev_g     : number of placeholders with empty values (?)
                        frnw_col      : number of columns in the fortran network map
                        paradim       : number of CN-mod parameters
                        mxnbathy_g    : maximum size of bathy data points
                        cwnrow_g      : crosswalk rows
                        cwncol_g      : crosswalk columns 

    usgs_da_reach_g: Array (integer) of indices of stream reaches where DA is applied 

    frnw_ar_g: Array (integer) for network mapping matrix, dimension nreach x frnw_col

    size_bathy_g: Array (integer) for number of bathy data points of each cross section, dimension nreach x maxNodes

    dbcd_g: Array (double) for downstream boundary data for number of timesteps, dimension being the latter

    para_ar_g: Array (double) for CN-mod parameters (e.g., Courant number etc); dimension a priori set
    
    diff_buff: Array (double) for diffusion parameters with dimensions:

                - Axis 0: nodes on reach
                - Axis 1: reach
                - Axis 2: Parameters (input into diffusion) in the following order:

                        - z_ar_g
                        - bo_ar_g
                        - traps_ar_g
                        - tw_ar_g
                        - twcc_ar_g     
                        - mann_ar_g
                        - manncc_ar_g
                        - dx_ar_g
                        - rdx_ar_g
                        - iniq
                        - z_thalweg_g

    so_ar_g: Array (double) for bottom slope (m/m) with dimensions (nodes on reach x reaches)

    ubcd_g: Array (double) for upstream boundary flow data (m3/sec) with dimensions (upstream time steps) x (nreach)

    qtrib_g: Array (double) for tributary time series data with dimensions (tributary time steps) x (nreach)

    usgs_da_g: Array (double) for usgs oberved streamflow data with dimensions (number of DA timesteps) x (nreach)

    qlat_g: Array (double) for lateral flow with dimensions (timesteps for lateral flow) x (nodes on reach) x (reaches)

    bathy_buf: Array (double) for bathymmetry parameters with dimensions (maximum size of bathy data points) x (nodes on reach) x (reaches):

                - Axis 0: bathy data point
                - Axis 1: nodes on reach
                - Axis 2: reach
                - Axis 3: Parameters (input into diffusion) in the following order:

                        - x_bathy_g
                        - z_bathy_g
                        - mann_bathy_g

    crosswalk_g: crosswalk array, crosswalk rows x columns  

    output_buf: Array (double) for qvd outputs:

                - Axis 0: ntss_ev_g (number of empty placeholders)
                - Axis 1: nodes on reach
                - Axis 2: reach
                - Axis 3: parameters in the following order:

                        - q_ev_g
                        - elv_ev_g
                        - depth_ev_g   

    '''

    # combine output: q_ev_g, elv_ev_g, depth_ev_g
    cdef reach.QVD_diff rv
    cdef reach.QVD_diff *out = &rv

    cdef:
        # entries in mem_size_buff
        int mxncomp_g, nrch_g, nts_ql_g, nts_ub_g, nts_db_g, nts_qtrib_g, nts_da_g, ntss_ev_g, frnw_col, paradim, mxnbathy_g, cwnrow_g, cwncol_g
        # entries in diff_buff
        float z_ar_g, bo_ar_g, traps_ar_g, tw_ar_g, twcc_ar_g, mann_ar_g, manncc_ar_g, dx_ar_g, rdx_ar_g, iniq, z_thalweg_g
        # entries in bathy buf 
        float x_bathy_g, z_bathy_g, mann_bathy_g


    # assign mem_size_buf (array dims)
    mxncomp_g = mem_size_buf[0]
    nrch_g = mem_size_buf[1]
    nts_ql_g = mem_size_buf[2]
    nts_ub_g = mem_size_buf[3]
    nts_db_g = mem_size_buf[4]
    nts_qtrib_g = mem_size_buf[5]
    nts_da_g = mem_size_buf[6]
    ntss_ev_g = mem_size_buf[7]
    frnw_col = mem_size_buf[8]
    paradim = mem_size_buf[9]
    mxnbathy_g = mem_size_buf[10]
    cwnrow_g = mem_size_buf[11]
    cwncol_g = mem_size_buf[12]


    # assign diffusion buffer
    for i in range(mxncomp_g):
        for j in range (nrch_g):

            z_ar_g[i,j] = diff_buff[i,j,0]
            bo_ar_g[i,j] = diff_buff[i,j,1]
            traps_ar_g[i,j] = diff_buff[i,j,2]
            tw_ar_g[i,j] = diff_buff[i,j,3]
            twcc_ar_g[i,j] = diff_buff[i,j,4]
            mann_ar_g[i,j] = diff_buff[i,j,5]
            manncc_ar_g[i,j] = diff_buff[i,j,6]
            dx_ar_g[i,j] = diff_buff[i,j,7]
            rdx_ar_g[i,j] = diff_buff[i,j,8]
            iniq[i,j] = diff_buff[i,j,9]
            z_thalweg_g[i,j] = diff_buff[i,j,10]


    # assign bathy buffer
    for k in range(mxnbathy_g):
        for i in range(mxncomp_g):
            for j in range (nrch_g):

                x_bathy_g[k,i,j] = bathy_buf[k,i,j,0]
                z_bathy_g[k,i,j] = bathy_buf[k,i,j,1]
                mann_bathy_g[k,i,j] = bathy_buf[k,i,j,2]

    # Call the diffusion wrapper; calls indirectly c_diffnw in pydiffusive.f90 
    reach.diffusion_callWrapper (
                            timestep_ar_g,
                            nts_ql_g, 
                            nts_ub_g, 
                            nts_db_g, 
                            ntss_ev_g, 
                            nts_qtrib_g, 
                            nts_da_g,
                            mxncomp_g, 
                            nrch_g, 
                            z_ar_g, 
                            bo_ar_g, 
                            traps_ar_g, 
                            tw_ar_g, 
                            twcc_ar_g, 
                            mann_ar_g, 
                            manncc_ar_g, 
                            so_ar_g, 
                            dx_ar_g, 
                            iniq, 
                            frnw_col, 
                            frnw_ar_g, 
                            qlat_g, 
                            ubcd_g, 
                            dbcd_g, 
                            qtrib_g, 
                            paradim, 
                            para_ar_g, 
                            mxnbathy_g, 
                            x_bathy_g, 
                            z_bathy_g, 
                            mann_bathy_g, 
                            size_bathy_g, 
                            usgs_da_g, 
                            usgs_da_reach_g, 
                            rdx_ar_g, 
                            cwnrow_g, 
                            cwncol_g, 
                            crosswalk_g, 
                            z_thalweg_g,
                            out 
                        )

                        # Variables definitions for calls to c_diffnw 
                        # integer(c_int), intent(in) :: nts_ql_g, nts_ub_g, nts_db_g, nts_qtrib_g, nts_da_g
                        # integer(c_int), intent(in) :: ntss_ev_g
                        # integer(c_int), intent(in) :: mxncomp_g, nrch_g
                        # integer(c_int), intent(in) :: frnw_col
                        # integer(c_int), intent(in) :: paradim
                        # integer(c_int), intent(in) :: mxnbathy_g
                        # integer(c_int), intent(in) :: cwnrow_g
                        # integer(c_int), intent(in) :: cwncol_g
                        # integer(c_int), dimension(nrch_g), intent(in) :: usgs_da_reach_g
                        # integer(c_int), dimension(nrch_g, frnw_col),    intent(in) :: frnw_ar_g
                        # integer(c_int), dimension(mxncomp_g, nrch_g),   intent(in) :: size_bathy_g 
                        # real(c_double), dimension(nts_db_g),            intent(in) :: dbcd_g
                        # real(c_double), dimension(paradim),             intent(in) :: para_ar_g
                        # real(c_double), dimension(:),                   intent(in) :: timestep_ar_g(9)
                        # real(c_double), dimension(mxncomp_g, nrch_g),   intent(in) :: z_ar_g, bo_ar_g, traps_ar_g
                        # real(c_double), dimension(mxncomp_g, nrch_g),   intent(in) :: tw_ar_g, twcc_ar_g
                        # real(c_double), dimension(mxncomp_g, nrch_g),   intent(in) :: mann_ar_g, manncc_ar_g
                        # real(c_double), dimension(mxncomp_g, nrch_g),   intent(inout) :: so_ar_g
                        # real(c_double), dimension(mxncomp_g, nrch_g),   intent(in) :: dx_ar_g, rdx_ar_g
                        # real(c_double), dimension(mxncomp_g, nrch_g),   intent(in) :: iniq
                        # real(c_double), dimension(mxncomp_g, nrch_g),   intent(in) :: z_thalweg_g
                        # real(c_double), dimension(nts_ub_g, nrch_g),    intent(in) :: ubcd_g
                        # real(c_double), dimension(nts_qtrib_g, nrch_g), intent(in) :: qtrib_g
                        # real(c_double), dimension(nts_da_g,    nrch_g), intent(in) :: usgs_da_g     
                        # real(c_double), dimension(nts_ql_g, mxncomp_g, nrch_g),   intent(in) :: qlat_g
                        # real(c_double), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in ) :: x_bathy_g
                        # real(c_double), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in ) :: z_bathy_g
                        # real(c_double), dimension(mxnbathy_g, mxncomp_g, nrch_g), intent(in ) :: mann_bathy_g
                        # real(c_double), dimension(cwnrow_g, cwncol_g),            intent(in ) :: crosswalk_g 
                        # real(c_double), dimension(ntss_ev_g, mxncomp_g, nrch_g),  intent(out) :: q_ev_g, elv_ev_g, depth_ev_g   

    # return results
    for k in range(ntss_ev_g):
        for i in range(mxncomp_g):
            for j in range (nrch_g):

                output_buf[k,i,j,0] = out.q_ev_g[k,i,j]
                output_buf[k,i,j,1] = out.elv_ev_g[k,i,j]
                output_buf[k,i,j,2] = out.depth_ev_g[k,i,j]




cpdef object compute_diffusive_structured(
    dict diff_ins
    ):
    
    """
    Compute network
    Args:
        diff_ins: diffusive input variables dictionary
    Notes:

    """

    # unpack/declare diffusive input variables
    cdef:
        double[::1] timestep_ar_g = np.asfortranarray(diff_ins['timestep_ar_g'])
        int nts_ql_g = diff_ins["nts_ql_g"]
        int nts_ub_g = diff_ins["nts_ub_g"]
        int nts_db_g = diff_ins["nts_db_g"]
        int ntss_ev_g = diff_ins["ntss_ev_g"] 
        int nts_qtrib_g = diff_ins['nts_qtrib_g']
        int nts_da_g = diff_ins["nts_da_g"]       
        int mxncomp_g = diff_ins["mxncomp_g"]
        int nrch_g = diff_ins["nrch_g"]
        double[::1,:] z_ar_g = np.asfortranarray(diff_ins["z_ar_g"])
        double[::1,:] bo_ar_g = np.asfortranarray(diff_ins["bo_ar_g"])
        double[::1,:] traps_ar_g = np.asfortranarray(diff_ins["traps_ar_g"])
        double[::1,:] tw_ar_g = np.asfortranarray(diff_ins["tw_ar_g"])
        double[::1,:] twcc_ar_g = np.asfortranarray(diff_ins["twcc_ar_g"])
        double[::1,:] mann_ar_g = np.asfortranarray(diff_ins["mann_ar_g"])
        double[::1,:] manncc_ar_g = np.asfortranarray(diff_ins["manncc_ar_g"])
        double[::1,:] so_ar_g = np.asfortranarray(diff_ins["so_ar_g"])
        double[::1,:] dx_ar_g = np.asfortranarray(diff_ins["dx_ar_g"])
        double[::1,:] iniq = np.asfortranarray(diff_ins["iniq"])
        int frnw_col = diff_ins["frnw_col"]
        int[::1,:] frnw_g = np.asfortranarray(diff_ins["frnw_g"])
        double[::1,:,:] qlat_g = np.asfortranarray(diff_ins["qlat_g"])
        double[::1,:] ubcd_g = np.asfortranarray(diff_ins["ubcd_g"])
        double[::1] dbcd_g = np.asfortranarray(diff_ins["dbcd_g"])
        double[::1,:] qtrib_g = np.asfortranarray(diff_ins["qtrib_g"])
        int paradim = diff_ins['paradim']
        double[::1] para_ar_g = np.asfortranarray(diff_ins["para_ar_g"])
        int mxnbathy_g = diff_ins['mxnbathy_g']
        double[::1,:,:] x_bathy_g = np.asfortranarray(diff_ins["x_bathy_g"])
        double[::1,:,:] z_bathy_g = np.asfortranarray(diff_ins["z_bathy_g"])
        double[::1,:,:] mann_bathy_g = np.asfortranarray(diff_ins["mann_bathy_g"])
        int[::1,:] size_bathy_g = np.asfortranarray(diff_ins["size_bathy_g"])    
        double[::1,:] usgs_da_g = np.asfortranarray(diff_ins["usgs_da_g"])   
        int[::1] usgs_da_reach_g = np.asfortranarray(diff_ins["usgs_da_reach_g"]) 
        double[::1,:] rdx_ar_g = np.asfortranarray(diff_ins["rdx_ar_g"])
        int cwnrow_g = diff_ins["cwnrow_g"]
        int cwncol_g = diff_ins["cwncol_g"]
        double[::1,:] crosswalk_g = np.asfortranarray(diff_ins["crosswalk_g"]) 
        double[::1,:] z_thalweg_g = np.asfortranarray(diff_ins["z_thalweg_g"])


    #
    # DEFINE BUFFERS 
    #
    # define array sizes buffer
    cdef int[:] mem_size_buf
    mem_size_buf = np.zeros(13, dtype='int32')
    #
    # define diffusion buffer
    cdef float[:,:,:] diff_buff
    diff_buff = np.zeros( (mxncomp_g, nrch_g, 11), dtype='float32')
    #
    # define bathy buffer
    cdef float[:,:,:,:] bathy_buf
    bathy_buf = np.zeros( (mxnbathy_g, mxncomp_g, nrch_g, 3), dtype='float32')
    #
    # define output buffer
    cdef float[:,:,:,:] bathy_buf
    bathy_buf = np.zeros( (ntss_ev_g, mxncomp_g, nrch_g, 3), dtype='float32')
    #

    # BEGINNING OF ORIGINAL DIFFUSIVE BRANCH

    #
    # set buffers to pass to the diffusion compute kernel
    #
    # assign array sizes buffer
    mem_size_buf[0] = mxncomp_g    # : maximum number of nodes in a single reach
    mem_size_buf[1] = nrch_g       # : number of reaches in the network
    mem_size_buf[2] = nts_ql_g     # : number of qlateral timesteps
    mem_size_buf[3] = nts_ub_g     # : upstream boundary time steps
    mem_size_buf[4] = nts_db_g     # : number of timesteps in downstream boundary data
    mem_size_buf[5] = nts_qtrib_g  # : number of timesteps in tributary time series data
    mem_size_buf[6] = nts_da_g     # : number of DA time steps
    mem_size_buf[7] = ntss_ev_g    # : number of placeholders with empty values (?)
    mem_size_buf[8] = frnw_col     # : number of columns in the fortran network map
    mem_size_buf[9] = paradim      # : number of CN-mod parameters
    mem_size_buf[10] = mxnbathy_g  # : maximum size of bathy data points
    mem_size_buf[11] = cwnrow_g    # : crosswalk rows
    mem_size_buf[12] = cwncol_g    # : crosswalk columns                 
    #
    # assign diffusion buffer
    for jj in range (nrch_g):
        for seg in range(n_nodes_reach):

            diff_buff[seg,jj,0] = z_ar_g[seg,jj]
            diff_buff[seg,jj,1]= bo_ar_g[seg,jj]
            diff_buff[seg,jj,2] = traps_ar_g[seg,jj]
            diff_buff[seg,jj,3] = tw_ar_g[seg,jj]
            diff_buff[seg,jj,4] = twcc_ar_g[seg,jj]
            diff_buff[seg,jj,5] = mann_ar_g[seg,jj]
            diff_buff[seg,jj,6] = manncc_ar_g[seg,jj]
            diff_buff[seg,jj,7] = dx_ar_g[seg,jj]
            diff_buff[seg,jj,8] = rdx_ar_g[seg,jj]
            diff_buff[seg,jj,9] = iniq[seg,jj]
            diff_buff[seg,jj,10] = z_thalweg_g[seg,jj]
    #
    # assign bathy buffer
    for jj in range (nrch_g):
        for seg in range(n_nodes_reach):           
            for kk in range(mxnbathy_g):
                        
                bathy_buf[kk,seg,jj,0] = x_bathy_g[kk,seg,jj]
                bathy_buf[kk,seg,jj,1] = z_bathy_g[kk,seg,jj]
                bathy_buf[kk,seg,jj,2] = mann_bathy_g[kk,seg,jj]


    # Other variables to pass:
    # timestep_ar_g: Array (float) of times and timesteps, as follows:
    #        dtini           = timestep_ar_g(1) ! initial timestep duration [sec]
    #        t0              = timestep_ar_g(2) ! simulation start time [hr]
    #        tfin            = timestep_ar_g(3) ! simulation end time [hr]
    #        saveInterval    = timestep_ar_g(4) ! output recording interval [sec]
    #        dt_ql           = timestep_ar_g(5) ! lateral inflow data time step [sec]
    #        dt_ub           = timestep_ar_g(6) ! upstream boundary time step [sec]
    #        dt_db           = timestep_ar_g(7) ! downstream boundary time step [sec]
    #        dt_qtrib        = timestep_ar_g(8) ! tributary data time step [sec]
    #        dt_da           = timestep_ar_g(9) ! data assimilation data time step [sec]

    # usgs_da_reach_g: Array (integer) of indices of stream reaches where DA is applied 
    #                   integer, dimension(nrch_g)

    #frnw_ar_g: Array (integer) for network mapping matrix, dimension nreach x frnw_col
    #                   integer, dimension(nrch_g, frnw_col)              

    #size_bathy_g: Array (integer) for number of bathy data points of each cross section, dimension nreach x maxNodes
    #                   integer, dimension(mxncomp_g, nrch_g)

    #dbcd_g: Array (double) for downstream boundary data for number of timesteps, dimension being the latter
    #                   float, dimension(nts_db_g)

    #para_ar_g: Array (double) for CN-mod parameters (e.g., Courant number etc); dimension a priori set
    #                   float, dimension(paradim)

    #so_ar_g: Array (double) for bottom slope (m/m) with dimensions (nodes on reach x reaches)
    #                   float, dimension(mxncomp_g, nrch_g)

    #ubcd_g: Array (double) for upstream boundary flow data (m3/sec) with dimensions (upstream time steps) x (nreach)
    #                   float, dimension(nts_ub_g, nrch_g)        

    #qtrib_g: Array (double) for tributary time series data with dimensions (tributary time steps) x (nreach)
    #                   float, dimension(nts_qtrib_g, nrch_g)

    #usgs_da_g: Array (double) for usgs oberved streamflow data with dimensions (number of DA timesteps) x (nreach)
    #                   float, dimension(nts_da_g, nrch_g)

    #qlat_g: Array (double) for lateral flow with dimensions (timesteps for lateral flow) x (nodes on reach) x (reaches)
    #                   float, dimension(nts_ql_g, mxncomp_g, nrch_g)

    #crosswalk_g: crosswalk array, crosswalk rows x columns  
    #                   float, dimension(cwnrow_g, cwncol_g)


    # call to compute diff kernel
    compute_diff_kernel(const float[:] timestep_ar_g, const int[:] mem_size_buf, const int[:] usgs_da_reach_g, const int[:,:] frnw_ar_g, const int[:,:] size_bathy_g, const float[:] dbcd_g, const float[:] para_ar_g, const float[:,:,:] diff_buff, float[:,:] so_ar_g, const float[:,:] ubcd_g, const float[:,:] qtrib_g, const float[:,:] usgs_da_g, const float[:,:,:] qlat_g, const float[:,:,:,:] bathy_buf, const float[:,:] crosswalk_g, float[:,:,:,:] output_buf) nogil:

    # Retrieve output (flow vel depth)
    for jj in range (nrch_g):
        for seg in range(mxncomp_g):        
            for kk in range(ntss_ev_g):
                        
                out.q_ev_g[kk,seg,jj] = output_buf[kk,seg,jj,0] 
                out.elv_ev_g[kk,seg,jj] = output_buf[kk,seg,jj,1]
                out.depth_ev_g[kk,seg,jj] = output_buf[kk,seg,jj,2] 

    # END OF DIFFUSIVE BRANCH

    # TODO: RETURN FORMAT
    return out
    
