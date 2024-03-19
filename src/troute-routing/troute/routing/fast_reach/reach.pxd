cdef struct QVD:
    float qdc
    float velc
    float depthc
    float cn
    float ck
    float X

cdef struct QVD_diff:
    float q_ev_g
    float elv_ev_g
    float depth_ev_g

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
        QVD *rv) nogil

cdef void wavediffusion(
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
        QVD_diff *rv) nogil 

cpdef float[:,:] compute_reach(const float[:] boundary,
                                const float[:,:] previous_state,
                                const float[:,:] parameter_inputs,
                                float[:,:] output_buffer) nogil
