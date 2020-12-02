MYHOMEPATH=/home/heiland
MYPYPATH=/home/heiland/software/gits/mypys
export PYTHONPATH="$MYPYPATH/mat_lib_plots:$MYPYPATH/sadptprj_riclyap_adi"  # $MYPYPATH/dolfin_navier_scipy"
export PYTHONPATH="$PYTHONPATH:$MYHOMEPATH/work/code/lqgbt-oseen"

RE=60
# CYLDIM=3
INIPERTURB=0.0
TRUNCAT=0.1
FBTYPE=-1  # no feedback
FBTYPE=2  # full state feedback
FBTYPE=1  # lqg-bt feedback
FBTYPE=4  # hinf-bt feedback
# FBTYPE=1  # lqg-bt feedback
PYMESS=1
# NUMPICARDS=6
SCALETEST=.04
NTS=25000
GRAMSFILE='/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/cylinderwake_re50_hinf.mat%outRegulator.Z%outFilter.Z%gam'
GRAMSFILE='/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/cylinderwake_re20_lqg.mat%Z_LQG_regulator%Z_LQG_filter'
GRAMSFILE='/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/cylinderwake_re20_hinf.mat%outRegulator.Z%outFilter.Z%gam'

# LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

echo tail -f $LOGFILE
python3 compscript.py --iniperturb=${INIPERTURB} --RE=${RE} \
    --closed_loop=${FBTYPE} --pymess \
    --scaletest=${SCALETEST} --truncat=${TRUNCAT} --strtogramfacs=${GRAMSFILE} \
    --Nts=${NTS}
# --cyldim=${CYLDIM}  # >> $LOGFILE
