import numpy as np
import argparse

from lqgbt_oseen import lqgbt_lnse
import datetime

problem = 'cylinderwake'
meshlevel = 1
problem = 'dbrotcyl'
meshlevel = 2

plotit = True
plotit = False
paraoutput = True
paraoutput = False

ddir = '/scratch/tbd/dnsdata/'
pymess = True
pymess = False
# relist = [None, 3e1, 4e1, 6e1]
# relist = [None, 30., 35., 40., 45., 50.]  # , 55.]
relist = [None, 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.]
# relist = [40., 40., 60.]
max_re_only = True  # consider only the last Re for the simu
max_re_only = False

# the input regularization parameter
gamma = 1e-0  # e5
# mesh parameter for the cylinder meshes
# whether to do bccontrol or distributed
bccontrol = True
palpha = 1e-5  # parameter for the Robin penalization

# ## TODO: remove this
cyldim = 3
simucyldim = 3  # the dim model used in the simulation

# where to truncate the LQGBT characteristic values
ctrunc = 1e-3  # , 1e-2, 1e-1, 1e-0]
# dimension of in and output spaces
NU = 'bcc'
# to what extend we perturb the initial value
perturbpara = 0*1e-5
# whether we use a perturbed system
trytofail = True
trytofail = False
ttf_npcrdstps = 6

# closed loop def
closed_loop = 'full_state_fb'
closed_loop = 'red_output_fb'
closed_loop = 'hinf_red_output_fb'
closed_loop = None
closed_loop = False
# what inival
whichinival = 'sstokes'  # steady state Stokes solution
whichinival, tpp = 'sstokes++', .5  # a developed state starting from sstokes
whichinival, tpp = 'snse+d++', 2.  # a developed state starting from sstokes
whichinival = 'sstate+d'  # sstate plus perturbation
tpp is tpp if whichinival == 'sstokes++' or whichinival == 'snse+d++' else None
# number of time steps -- also define the lag in the control application
addinputd = True  # whether to add disturbances through the input
duampltd = 1e-6

scaletest = 50.  # for 1. we simulate till 12.
baset0, basetE, baseNts = 0.0, 12.0, 12*2**8
dudict = dict(addinputd=addinputd, ta=0., tb=1., ampltd=duampltd,
              uvec=np.array([1, 1]).reshape((2, 1)))

parser = argparse.ArgumentParser()
parser.add_argument("--RE", type=float, help="Reynoldsnumber")
parser.add_argument("--RE_ini", type=float, help="Re for initialization")
parser.add_argument("--pymess", help="Use pymess", action='store_true')
parser.add_argument("--ttf", help="trytofail", action='store_true')
parser.add_argument("--ttf_npcrdstps", type=int,
                    help="Whether/when to break the Picard/Newton iteration",
                    choices=range(40), default=ttf_npcrdstps)
parser.add_argument("--tE", type=float,
                    help="final time of the simulation", default=basetE)
parser.add_argument("--Nts", type=float,
                    help="number of time steps", default=baseNts)
parser.add_argument("--mesh", type=int,
                    help="mesh parameter", default=meshlevel)
parser.add_argument("--scaletest", type=float,
                    help="scale the test size", default=scaletest)
parser.add_argument("--truncat", type=float,
                    help="truncation threshhold for BTs", default=ctrunc)
parser.add_argument("--strtogramfacs", type=str,
                    help="info where the gramian factors are stored\n" +
                    "use like 'fname%zwc.adress%zwo.address[%gamma.address]'")
parser.add_argument("--iniperturb", type=float,
                    help="magnitude for the perturbation of the initial value")
parser.add_argument("--closed_loop", type=int, choices=[-1, 0, 1, 2, 4],
                    help="-1: None,\n0: False,\n 1: 'red_output_fb'," +
                    "\n 2:'full_output_fb',\n 4: 'hinf_red_output_fb'")
parser.add_argument("--problem", type=str,
                    choices=['cylinderwake', 'dbrotcyl'],
                    help="which setup to consider", default=problem)
parser.add_argument("--shortlogfile", type=str, default='',
                    help="name of a file where short information is store")
args = parser.parse_args()
print(args)

if args.ttf:
    trytofail = True
if args.RE is not None:
    relist = [args.RE_ini, args.RE]
closedloopdct = {-1: None, 0: False, 1: 'red_output_fb',
                 2: 'full_output_fb', 4: 'hinf_red_output_fb'}
if args.closed_loop is not None:
    closed_loop = closedloopdct[args.closed_loop]
if args.iniperturb is not None:
    whichinival = 'sstate+d'  # override whichinival
    perturbpara = np.float(args.iniperturb)
if args.pymess:
    pymess = True

t0, tE, Nts = 0.0, args.scaletest*args.tE, np.int(args.scaletest*args.Nts)

if closed_loop == 'hinf_red_output_fb':
    closed_loop = 'red_output_fb'
    hinf = True
else:
    hinf = False

if max_re_only:
    relist = relist[-2:]

if args.problem == 'dbrotcyl':
    meshprfx = 'mesh/2D-double-rotcyl-meshes/2D-double-rotcyl'
    geodata = 'mesh/2D-double-rotcyl-meshes/2D-double-rotcyl_geo_cntrlbc.json'
    shortname = 'drc'
    Cgrid = (4, 1)  # grid of the sensors -- defines the C
elif args.problem == 'cylinderwake':
    meshprfx = 'mesh/2D-outlet-meshes/karman2D-outlets'
    geodata = 'mesh/2D-outlet-meshes/karman2D-outlets_geo_cntrlbc.json'
    shortname = 'cw'
    Cgrid = (3, 1)  # grid of the sensors -- defines the C

meshfile = meshprfx + '_lvl{0}.xml.gz'.format(args.mesh)
physregs = meshprfx + '_lvl{0}_facet_region.xml.gz'.format(args.mesh)

# print reynolds number and discretization lvl
infostring = ('Problem        = {0}'.format(problem) +
              '\nRe             = {0}'.format(relist) +
              '\nmeshlevel      = {0}'.format(args.mesh) +
              '\npymess         = {0}'.format(args.pymess) +
              '\nclosed_loop    = {0}'.format(closed_loop) +
              '\nH_infty        = {0}'.format(hinf) +
              '\ntrunc at       = {0}'.format(args.truncat) +
              '\nini_perturb    = {0}'.format(perturbpara) +
              '\nsys_perturb    = {0}'.format(trytofail) +
              '\nttf_npcrdstps  = {0}'.format(args.ttf_npcrdstps) +
              '\nt0, tE, Nts    = {0}, {1}, {2}'.format(t0, tE, Nts) +
              '\nu_d: ta, tb, A = ' +
              '{0}, {1}, {2}\n'.format(dudict['ta'], dudict['tb'],
                                       dudict['ampltd'])
              )

print(infostring)

if pymess:
    nwtn_adi_dict = dict(verbose=True, maxit=45, aditol=1e-8,
                         nwtn_res2_tol=4e-8, linesearch=True)
else:
    nwtn_adi_dict = dict(adi_max_steps=450,  # 450,
                         adi_newZ_reltol=2e-8,
                         nwtn_max_steps=30,
                         nwtn_upd_reltol=2e-8,
                         nwtn_upd_abstol=1e-7,
                         ms=[-100., -50., -10., -2.0, -1.3,
                             -1.0, -0.9, -0.5],
                         # ms=[-10., -2.0, -1.3, -1.0, -0.9, -0.5],
                         verbose=True,
                         full_upd_norm_check=False,
                         check_lyap_res=False)

logstr = 'logs/log_cyldim{0}NU{1}C{2[0]}{2[1]}gamma{3}'.\
    format(cyldim, NU, Cgrid, gamma) +\
    'closedloop{0}'.format(closed_loop) +\
    't0{0}tE{1}Nts{2}'.format(t0, tE, Nts) +\
    'Re{2}to{3}kappa{0}to{1}eps{4}'.format(ctrunc, ctrunc,
                                           relist[0], relist[-1], perturbpara)

# if logtofile:
#     print('log goes ' + logstr)
#     print('how about \ntail -f ' + logstr)
#     sys.stdout = open(logstr, 'a', 0)

print(('{0}'*10 + '\n log started at {1} \n' + '{0}'*10).
      format('X', str(datetime.datetime.now())))

simudict = dict(meshparams=dict(strtomeshfile=meshfile,
                                strtophysicalregions=physregs,
                                strtobcsobs=geodata),
                problemname=args.problem, shortname=shortname,
                NU=NU, Cgrid=Cgrid,
                trunc_lqgbtcv=args.truncat,
                t0=t0, tE=tE, Nts=Nts,
                nwtn_adi_dict=nwtn_adi_dict,
                paraoutput=paraoutput, multiproc=True,
                pymess=args.pymess,
                bccontrol=bccontrol, gamma=gamma,
                plotit=plotit,
                ddir=ddir,
                whichinival=whichinival, tpp=tpp,
                dudict=dudict,
                hinf=hinf,
                strtogramfacs=args.strtogramfacs,
                trytofail=trytofail, ttf_npcrdstps=args.ttf_npcrdstps,
                closed_loop=closed_loop,
                perturbpara=perturbpara)

for cre in range(1, len(relist)):
    simudict.update(dict(use_ric_ini=relist[cre-1], Re=relist[cre]))
    if not closed_loop:
        lqgbt_lnse.lqgbt(**simudict)
    else:
        ffflag, ymys, ystr = lqgbt_lnse.lqgbt(**simudict)
        if not args.shortlogfile == '':
            with open(args.shortlogfile, 'a') as slfl:
                slfl.write(infostring)
                slfl.write('\n****** RESULT---> ******\n')
                if ffflag == 1:
                    slfl.write('failed: ' + ystr)
                else:
                    slfl.write('|dy|: {0}: '.format(ymys) + ystr)
                slfl.write('\n****** <---RESULT ******\n\n')
                slfl.write(infostring)
