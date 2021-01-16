MYHOMEPATH=/home/heiland
# MYPYPATH=/home/heiland/software/mypys
# export PYTHONPATH="$MYPYPATH/mat_lib_plots:$MYPYPATH/sadptprj_riclyap_adi"
export PYTHONPATH="$MYHOMEPATH/work/code/lqgbt-oseen"
# export PYTHONPATH="$PYTHONPATH:$MYPYPATH/dolfin_navier_scipy"
echo $PYTHONPATH

RE=60
NTS=3072  # 12*2**8
PROBLEM=dbrotcyl
MSHLVL=2
SHRTLF=resultsoverview-dbrotcyl.md

# RE=60
# NTS=20000

INIPERTURB=0.0
# TRUNCAT=.00015
TRUNCAT=.125
FBTYPE=2  # full state feedback
FBTYPE=1  # lqg-bt feedback
# FBTYPE=1  # lqg-bt feedback
PYMESS=1
FBTYPE=-1  # no feedback
FBTYPE=4  # hinf-bt feedback
REPERTURB=-15

SCALETEST=25.

GRAMSPATH=/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/
HNFQR=_hinf.mat%outRegulator.Z%outFilter.Z%gam
LQGQR=_lqg.mat%Z_LQG_regulator%Z_LQG_filter

GRAMSFILE=${GRAMSPATH}doublecylinder_re${RE}${LQGQR}
GRAMSFILE=${GRAMSPATH}doublecylinder_re${RE}${HNFQR}

# LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

python3 compscript.py \
    --problem=${PROBLEM} --mesh=${MSHLVL} \
    --iniperturb=${INIPERTURB} --RE=${RE} \
    --closed_loop=${FBTYPE} --pymess \
    --scaletest=${SCALETEST} --truncat=${TRUNCAT} \
    --strtogramfacs=${GRAMSFILE} \
    --Nts=${NTS} \
    --ttf_ptre --ttf_value=${REPERTURB} \
    --shortlogfile=${SHRTLF}
