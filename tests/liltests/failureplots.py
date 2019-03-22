from dolfin_navier_scipy.data_output_utils import plot_outp_sig

plot_outp_sig("data/cylinderwake_Re125.0_NV19512_bcc_NY3_lqgbt__lqgbtcv0.0001red_output_fbt00.0tE18.0Nts3601.0inipert1e-06", tikzstr='redfb', fignum=1)

plot_outp_sig("data/cylinderwake_Re125.0_NV19512_bcc_NY3_lqgbt_MAF_ttfnpcrds5__lqgbtcv0.0001red_output_fbt00.0tE18.0Nts3601.0inipert1e-06", tikzstr='corrredfb', fignum=2)

plot_outp_sig("data/cylinderwake_Re125.0_NV19512_bcc_NY3_lqgbt_MAF_ttfnpcrds5__lqgbtcv0.0001Nonet00.0tE18.0Nts3601.0inipert1e-06", tikzstr='nofb', fignum=3)

plot_outp_sig("data/cylinderwake_Re125.0_NV19512_bcc_NY3_lqgbt_MAF_ttfnpcrds5__lqgbtcv0.0001full_state_fbt00.0tE18.0Nts3601.0inipert1e-06", tikzstr='corrfullfb', fignum=4)
