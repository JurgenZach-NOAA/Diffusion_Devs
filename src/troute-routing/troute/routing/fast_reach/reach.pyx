import cython
#from libc.stdio cimport printf

from .fortran_wrappers cimport c_muskingcungenwm, c_diffnw

@cython.boundscheck(False)
cdef void muskingcunge(float dt,
        float qup,
        float quc,
        float qdp,
        float ql,
        float dx,
        float bw,
        float tw,
        float twcc,
        float n,
        float ncc,
        float cs,
        float s0,
        float velp,
        float depthp,
        QVD *rv) nogil:

    cdef:
        float qdc = 0.0
        float depthc = 0.0
        float velc = 0.0
        float ck = 0.0
        float cn = 0.0
        float X = 0.0

    #printf("reach.pyx before %3.9f\t", depthc)
    c_muskingcungenwm(
        &dt,
        &qup,
        &quc,
        &qdp,
        &ql,
        &dx,
        &bw,
        &tw,
        &twcc,
        &n,
        &ncc,
        &cs,
        &s0,
        &velp,
        &depthp,
        &qdc,
        &velc,
        &depthc,
        &ck,
        &cn,
        &X)
    #printf("reach.pyx after %3.9f\t", depthc)

    rv.qdc = qdc
    rv.depthc = depthc
    rv.velc = velc

    # to do: make these additional variable's conditional, somehow
    rv.ck = ck
    rv.cn = cn
    rv.X = X


'''
#const float[:] timestep_ar_g, const int[:] mem_size_buf, const int[:] usgs_da_reach_g, const int[:,:] frnw_ar_g, const int[:,:] size_bathy_g, const float[:] dbcd_g, const float[:] para_ar_g, const float[:,:,:] diff_buff, float[:,:] so_ar_g, const float[:,:] ubcd_g, const float[:,:] qtrib_g, const float[:,:] usgs_da_g, const float[:,:,:] qlat_g, const float[:,:,:,:] bathy_buf, const float[:,:] crosswalk_g, float[:,:,:,:] output_buf
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
'''


@cython.boundscheck(False)
cdef void diffusion_callWrapper(
        double timestep_ar_g,
        int nts_ql_g, 
        int nts_ub_g, 
        int nts_db_g, 
        int ntss_ev_g, 
        int nts_qtrib_g, 
        int nts_da_g,
        int mxncomp_g, 
        int nrch_g, 
        double z_ar_g, 
        double bo_ar_g, 
        double traps_ar_g, 
        double tw_ar_g, 
        double twcc_ar_g, 
        double mann_ar_g, 
        double manncc_ar_g, 
        double so_ar_g, 
        double dx_ar_g, 
        double iniq, 
        int frnw_col, 
        int frnw_ar_g, 
        double qlat_g, 
        double ubcd_g, 
        double dbcd_g, 
        double qtrib_g, 
        int paradim, 
        double para_ar_g, 
        int mxnbathy_g, 
        double x_bathy_g, 
        double z_bathy_g, 
        double mann_bathy_g, 
        int size_bathy_g, 
        double usgs_da_g, 
        int usgs_da_reach_g, 
        double rdx_ar_g, 
        int cwnrow_g, 
        int cwncol_g, 
        double crosswalk_g, 
        double z_thalweg_g,
        QVD_diff *rv) nogil:

    cdef:
        double q_ev_g = 0.0
        double elv_ev_g = 0.0
        double depth_ev_g = 0.0

    c_diffnw(
            &timestep_ar_g,
            &nts_ql_g,
            &nts_ub_g,
            &nts_db_g,
            &ntss_ev_g,
            &nts_qtrib_g,
            &nts_da_g,
            &mxncomp_g,
            &nrch_g,
            &z_ar_g,
            &bo_ar_g,
            &traps_ar_g,
            &tw_ar_g,
            &twcc_ar_g,
            &mann_ar_g,
            &manncc_ar_g,
            &so_ar_g,
            &dx_ar_g,
            &iniq,
            &frnw_col,
            &frnw_ar_g,
            &qlat_g,
            &ubcd_g,
            &dbcd_g,
            &qtrib_g,
            &paradim,
            &para_ar_g,
            &mxnbathy_g,
            &x_bathy_g,
            &z_bathy_g,
            &mann_bathy_g,
            &size_bathy_g,
            &usgs_da_g,
            &usgs_da_reach_g,
            &rdx_ar_g,
            &cwnrow_g,
            &cwncol_g,
            &crosswalk_g, 
            &z_thalweg_g,
            &q_ev_g,
            &elv_ev_g,
            &depth_ev_g)

    rv.q_ev_g = q_ev_g
    rv.elv_ev_g = elv_ev_g
    rv.depth_ev_g = depth_ev_g


cpdef dict compute_reach_kernel(float dt,
        float qup,
        float quc,
        float qdp,
        float ql,
        float dx,
        float bw,
        float tw,
        float twcc,
        float n,
        float ncc,
        float cs,
        float s0,
        float velp,
        float depthp):

    cdef QVD rv
    cdef QVD *out = &rv

    muskingcunge(
        dt,
        qup,
        quc,
        qdp,
        ql,
        dx,
        bw,
        tw,
        twcc,
        n,
        ncc,
        cs,
        s0,
        velp,
        depthp,
        out)

    return rv


cpdef long boundary_shape() nogil:
    return 2

cpdef long previous_state_cols() nogil:
    return output_buffer_cols()

cpdef long parameter_inputs_cols() nogil:
    return 13

cpdef long output_buffer_cols() nogil:
    return 3

@cython.boundscheck(False)
cpdef float[:,:] compute_reach(const float[:] boundary,
                                const float[:,:] previous_state,
                                const float[:,:] parameter_inputs,
                                float[:,:] output_buffer,
                                Py_ssize_t size=0) nogil:
    """
    Compute a reach

    Arguments:
        boundary: [qup, quc]
        previous_state: Previous state for each node in the reach [qdp, velp, depthp]
        parameter_inputs: Parameterization of the reach at node.
            qlat, dt, dx, bw, tw, twcc, n, ncc, cs, s0
        output_buffer: Current state [qdc, velc, depthc]

    """
    cdef QVD rv
    cdef QVD *out = &rv

    cdef:
        float dt, qlat, dx, bw, tw, twcc, n, ncc, cs, s0, qdp, velp, depthp
        Py_ssize_t i, rows

    # check that previous state, parameter_inputs and output_buffer all have same axis 0
    if size > 0:
        rows = size
        if (parameter_inputs.shape[0] < rows
                or output_buffer.shape[0] < rows
                or previous_state.shape[0] < rows):
            raise ValueError(f"axis 0 is not long enough for {size}")
    else:
        rows = previous_state.shape[0]
        if rows != parameter_inputs.shape[0] or rows != output_buffer.shape[0]:
            raise ValueError("axis 0 of input arguments do not agree")

    # check bounds
    if boundary.shape[0] < 2:
        raise IndexError
    if parameter_inputs.shape[1] < 10:
        raise IndexError
    if output_buffer.shape[1] < 3:
        raise IndexError
    if previous_state.shape[1] < 3:
        raise IndexError

    cdef float qup = boundary[0]
    cdef float quc = boundary[1]

    for i in range(rows):
        qlat = parameter_inputs[i, 0]
        dt = parameter_inputs[i, 1]
        dx = parameter_inputs[i, 2]
        bw = parameter_inputs[i, 3]
        tw = parameter_inputs[i, 4]
        twcc = parameter_inputs[i, 5]
        n = parameter_inputs[i, 6]
        ncc = parameter_inputs[i, 7]
        cs = parameter_inputs[i, 8]
        s0 = parameter_inputs[i, 9]

        qdp = previous_state[i, 0]
        velp = previous_state[i, 1]
        depthp = previous_state[i, 2]

        muskingcunge(
                    dt,
                    qup,
                    quc,
                    qdp,
                    qlat,
                    dx,
                    bw,
                    tw,
                    twcc,
                    n,
                    ncc,
                    cs,
                    s0,
                    velp,
                    depthp,
                    out)

        output_buffer[i, 0] = quc = out.qdc
        output_buffer[i, 1] = out.velc
        output_buffer[i, 2] = out.depthc

        qup = qdp
    return output_buffer
