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

from troute.network.diffusive.diffusive_reach cimport Diff_Segment, Diff_Reach, _Diff_Segment, get_diffusive_segment

cimport troute.routing.fast_reach.reach as reach

@cython.boundscheck(False)
# Equivalent of compute_reach_kernel in MC case
cdef void compute_diff_kernel(const float[:] timestep_ar_g, const int[:] mem_size_buf, const int[:] usgs_da_reach_g, const int[:,:] frnw_ar_g, const int[:,:] size_bathy_g, const float[:] dbcd_g, const float[:] para_ar_g, const float[:,:,:] diff_buff, float[:,:] so_ar_g, const float[:,:] ubcd_g, const float[:,:] qtrib_g, const float[:,:] usgs_da_g, const float[:,:,:] qlat_g, const float[:,:,:,:] bathy_buf, const float[:,:] crosswalk_g, float[:,:,:,:] output_buf) nogil:


    """
    
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

    """

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
    reach.wavediffusion (
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







cpdef object column_mapper(object src_cols):
    """Map source columns to columns expected by algorithm"""
    cdef object index = {}
    cdef object i_label
    for i_label in enumerate(src_cols):
        index[i_label[1]] = i_label[0]

    cdef object rv = []
    cdef object label
    #qlat, dt, dx, bw, tw, twcc, n, ncc, cs, s0, qdp, velp, depthp
    for label in ['dt', 'dx', 'bw', 'tw', 'twcc', 'n', 'ncc', 'cs', 's0']:
        rv.append(index[label])
    return rv





cpdef object compute_network_structured(
    int nsteps,
    float dt,
    int qts_subdivisions,
    list reaches_wTypes, # a list of tuples
    dict upstream_connections,
    const long[:] data_idx,
    object[:] data_cols,
    const float[:,:] data_values,
    const float[:,:] initial_conditions,
    const float[:,:] qlat_values,
    list lake_numbers_col,
    const double[:,:] wbody_cols,
    dict data_assimilation_parameters,
    const int[:,:] reservoir_types,
    bint reservoir_type_specified,
    str model_start_time,
    const float[:,:] usgs_values,
    const int[:] usgs_positions,
    const int[:] usgs_positions_reach,
    const int[:] usgs_positions_gage,
    const float[:] lastobs_values_init,
    const float[:] time_since_lastobs_init,
    const double da_decay_coefficient,
    const float[:,:] reservoir_usgs_obs,
    const int[:] reservoir_usgs_wbody_idx,
    const float[:] reservoir_usgs_time,
    const float[:] reservoir_usgs_update_time,
    const float[:] reservoir_usgs_prev_persisted_flow,
    const float[:] reservoir_usgs_persistence_update_time,
    const float[:] reservoir_usgs_persistence_index,
    const float[:,:] reservoir_usace_obs,
    const int[:] reservoir_usace_wbody_idx,
    const float[:] reservoir_usace_time,
    const float[:] reservoir_usace_update_time,
    const float[:] reservoir_usace_prev_persisted_flow,
    const float[:] reservoir_usace_persistence_update_time,
    const float[:] reservoir_usace_persistence_index,
    const float[:,:] reservoir_rfc_obs,
    const int[:] reservoir_rfc_wbody_idx,
    const int[:] reservoir_rfc_totalCounts,
    list reservoir_rfc_file,
    const int[:] reservoir_rfc_use_forecast,
    const int[:] reservoir_rfc_timeseries_idx,
    const float[:] reservoir_rfc_update_time,
    const int[:] reservoir_rfc_da_timestep,
    const int[:] reservoir_rfc_persist_days,
    dict upstream_results={},
    bint assume_short_ts=False,
    bint return_courant=False,
    int da_check_gage = -1,
    bint from_files=True,
    dict diff_ins
    ):
    
    """
    Compute network
    Args:
        nsteps (int): number of time steps
        reaches_wTypes (list): List of tuples: (reach, reach_type), where reach_type is 0 for Muskingum Cunge reach and 1 is a reservoir
        upstream_connections (dict): Network
        data_idx (ndarray): a 1D sorted index for data_values
        data_values (ndarray): a 2D array of data inputs (nodes x variables)
        qlats (ndarray): a 2D array of qlat values (nodes x nsteps). The index must be shared with data_values
        initial_conditions (ndarray): an n x 3 array of initial conditions. n = nodes, column 1 = qu0, column 2 = qd0, column 3 = h0
        assume_short_ts (bool): Assume short time steps (quc = qup)
    Notes:
        Array dimensions are checked as a precondition to this method.
        This version creates python objects for segments and reaches,
        but then uses only the C structures and access for efficiency
    """

    #Make ndarrays from the mem views for convience of indexing...may be a better method
    cdef np.ndarray[float, ndim=2] data_array = np.asarray(data_values)
    cdef np.ndarray[float, ndim=2] init_array = np.asarray(initial_conditions)
    cdef np.ndarray[float, ndim=2] qlat_array = np.asarray(qlat_values)
    cdef np.ndarray[double, ndim=2] wbody_parameters = np.asarray(wbody_cols)
    
    ###### Declare/type variables #####
    # Source columns
    cdef Py_ssize_t[:] scols = np.array(column_mapper(data_cols), dtype=np.intp)
    cdef Py_ssize_t max_buff_size = 0
    #lists to hold reach definitions, i.e. list of ids
    cdef list reach
    cdef list upstream_reach
    #lists to hold segment ids
    cdef list segment_ids
    cdef list upstream_ids
    
    
    #buffers to pass to compute_reach_kernel
    cdef float[:,:] buf_view
    cdef float[:,:] out_buf
    cdef float[:] lateral_flows
    # list of reach objects to operate on
    cdef list reach_objects = []
    cdef list segment_objects



    cdef long sid
    cdef _MC_Segment segment
    #pr.enable()
    #Preprocess the raw reaches, creating MC_Reach/MC_Segments



    for reach, reach_type in reaches_wTypes:
        upstream_reach = upstream_connections.get(reach[0], ())
        upstream_ids = binary_find(data_idx, upstream_reach)

        segment_ids = binary_find(data_idx, reach)
        #Set the initial condtions before running loop
        flowveldepth_nd[segment_ids, 0] = init_array[segment_ids]
        segment_objects = []
        #Find the max reach size, used to create buffer for compute_reach_kernel
        if len(segment_ids) > max_buff_size:
            max_buff_size=len(segment_ids)

        for sid in segment_ids:
            #Initialize parameters  from the data_array, and set the initial initial_conditions
            #These aren't actually used (the initial contions) in the kernel as they are extracted from the
            #flowdepthvel array, but they could be used I suppose.  Note that velp isn't used anywhere, so
            #it is inialized to 0.0
            segment_objects.append(
            MC_Segment(sid, *data_array[sid, scols], init_array[sid, 0], 0.0, init_array[sid, 2])
        )
        reach_objects.append(
            #tuple of MC_Reach and reach_type
            MC_Reach(segment_objects, array('l',upstream_ids))
            )




    # template for cdef :
    #     cdef np.ndarray[int, ndim=1] usgs_idx  = np.asarray(reservoir_usgs_wbody_idx)

    

    
    cdef np.ndarray fill_index_mask = np.ones_like(data_idx, dtype=bool)
    cdef Py_ssize_t fill_index
    cdef long upstream_tw_id
    cdef dict tmp
    cdef int idx
    cdef float val

    for upstream_tw_id in upstream_results:
        tmp = upstream_results[upstream_tw_id]
        fill_index = tmp["position_index"]
        fill_index_mask[fill_index] = False
        for idx, val in enumerate(tmp["results"]):
            flowveldepth_nd[fill_index, (idx//qvd_ts_w) + 1, idx%qvd_ts_w] = val
            if data_idx[fill_index]  in lake_numbers_col:
                res_idx = binary_find(lake_numbers_col, [data_idx[fill_index]])
                flowveldepth_nd[fill_index, 0, 0] = wbody_parameters[res_idx, 9] # TODO ref dataframe column label
            else:
                flowveldepth_nd[fill_index, 0, 0] = init_array[fill_index, 0] # initial flow condition
                flowveldepth_nd[fill_index, 0, 2] = init_array[fill_index, 2] # initial depth condition



    # essential variable definitions
    nrch_g = num_reaches
    n_nodes_reach = r_diff.reach.diff_reach.num_segments






    timestep_ar_g = diff_ins["timestep_ar_g"]
    nts_ql_g = diff_ins["nts_ql_g"]
    nts_ub_g = diff_ins["nts_ub_g"]
    nts_db_g = diff_ins["nts_db_g"]
    nts_qtrib_g = diff_ins["nts_qtrib_g"]
    ntss_ev_g = diff_ins["ntss_ev_g"]
    nts_da_g = diff_ins["nts_da_g"]
    mxncomp_g = diff_ins["mxncomp_g"]
    nrch_g = diff_ins["nrch_g"]
    z_ar_g = diff_ins["z_ar_g"]
    bo_ar_g = diff_ins["bo_ar_g"] 
    traps_ar_g = diff_ins["traps_ar_g"] 
    tw_ar_g = diff_ins["tw_ar_g"] 
    twcc_ar_g = diff_ins["twcc_ar_g"] 
    mann_ar_g = diff_ins["mann_ar_g"]
    manncc_ar_g = diff_ins["manncc_ar_g"]
    so_ar_g = diff_ins["so_ar_g"]
    dx_ar_g = diff_ins["dx_ar_g"]
    frnw_col = diff_ins["frnw_col"]
    frnw_g = diff_ins["frnw_g"]
    qlat_g = diff_ins["qlat_g"]
    ubcd_g = diff_ins["ubcd_g"]
    dbcd_g = diff_ins["dbcd_g"]
    qtrib_g = diff_ins["qtrib_g"]
    paradim = diff_ins["paradim"]
    para_ar_g = diff_ins["para_ar_g"]
    mxnbathy_g = diff_ins["mxnbathy_g"] 
    x_bathy_g = diff_ins["x_bathy_g"] 
    z_bathy_g = diff_ins["z_bathy_g"]
    mann_bathy_g = diff_ins["mann_bathy_g"] 
    size_bathy_g = diff_ins["size_bathy_g"]
    iniq = diff_ins["iniq"]
    pynw = diff_ins["pynw"]
    ordered_reaches = diff_ins["ordered_reaches"]
    usgs_da_g = diff_ins["usgs_da_g"]
    usgs_da_reach_g = diff_ins["usgs_da_reach_g"]
    rdx_ar_g = diff_ins["rdx_ar_g"]
    crosswalk_nrow = diff_ins["cwnrow_g"]
    crosswalk_ncol = diff_ins["cwncol_g"]
    crosswalk_g = diff_ins["crosswalk_g"] 
    z_thalweg_g = diff_ins["z_thalweg_g"]

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

    '''
    # LEGACY ADAPTATION

    cdef int num_reaches = len(reach_objects)
    #Dynamically allocate a C array of reach structs
    cdef _DiffReach* reach_diff_structs = <_DiffReach*>malloc(sizeof(_DiffReach)*num_reaches)
    #Populate the above array with the structs contained in each reach object
    for i in range(num_reaches):
        reach_diff_structs[i] = (<DiffReach>reach_objects[i])._reach

    #reach iterator
    cdef _DiffReach* r_diff
    '''


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
    for _i in range(n_nodes_reach):
        seg = get_diffusive_segment(r_diff, _i)
        for jj in range (nrch_g):
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
    for _i in range(n_nodes_reach):
        seg = get_diffusive_segment(r_diff, _i)             
        for jj in range (nrch_g):     
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
    for _i in range(mxncomp_g):
        seg = get_diffusive_segment(r_diff, _i)                     
        for jj in range (nrch_g):
            for kk in range(ntss_ev_g):
                        
                out.q_ev_g[kk,seg,jj] = output_buf[kk,seg,jj,0] 
                out.elv_ev_g[kk,seg,jj] = output_buf[kk,seg,jj,1]
                out.depth_ev_g[kk,seg,jj] = output_buf[kk,seg,jj,2] 

    # END OF DIFFUSIVE BRANCH

    # TODO: RETURN FORMAT
