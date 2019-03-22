from dolfin_navier_scipy.data_output_utils import plot_outp_sig

lqgfignum, hinfignum = 101, 201
pdthnf = dict(notikz=True, fignum=hinfignum)
pdtlqg = dict(notikz=True, fignum=lqgfignum)
# LQG -- clearly unstable, hinf -- almost stable
# plot_outp_sig("data/cw90.01.0_93843-16rofb0.001ssd0.001maf3", **pdtlqg)
# plot_outp_sig("data/cw90.01.0_93843-16hinfrofb0.001ssd0.001maf3", **pdthnf)

# lqgfignum += 1
# hinfignum += 1
# # less perturbation -- similar picture, hinf instab but not visible
# plot_outp_sig("data/cw90.01.0_93843-16rofb0.001ssd0.0001maf3", **pdtlqg)
# plot_outp_sig("data/cw90.01.0_93843-16hinfrofb0.001ssd0.0001maf3", **pdthnf)

# # RE=100 -- very similar performance
# lqgfignum += 1
# hinfignum += 1
# plot_outp_sig("data/cw100.01.0_93843-16hinfrofb0.001ssd0.001maf4", **pdthnf)
# plot_outp_sig("data/cw100.01.0_93843-16rofb0.001ssd0.0001maf4", **pdtlqg)

# RE=90 / input disturbance only
# nps=3 -- hinfrofb not really stable
# lqgfignum += 1
# hinfignum += 1
# plot_outp_sig("data/cw90.01.0_93843-16hinfrofb0.01ssd0.0maf3", **pdthnf)
# plot_outp_sig("data/cw90.01.0_93843-16rofb0.01ssd0.0maf3", **pdtlqg)

# RE=90 / input disturbance only
# nps=3 -- less disturbance `ampltd=0.01` -- more truncation
# hinf stable / lqgbt not
tkzstr = 'cw90-N93843-trc5e-2-ssd0-ud1e-2'
lqgfignum += 1
hinfignum += 1
plot_outp_sig("data/cw90.01.0_93843-16hinfrofb0.05ssd0.0maf3",
              tikzstr=tkzstr+'hinfbt')
plot_outp_sig("data/cw90.01.0_93843-16rofb0.05ssd0.0maf3",
              tikzstr=tkzstr+'lqgbt')
plot_outp_sig("data/cw90.01.0_93843-16__ssd0.0maf3",
              tikzstr=tkzstr+'nofb')
