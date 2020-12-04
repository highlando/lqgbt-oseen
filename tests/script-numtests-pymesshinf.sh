MYHOMEPATH=/home/heiland
MYPYPATH=/home/heiland/software/gits/mypys
export PYTHONPATH="$MYPYPATH/mat_lib_plots:$MYPYPATH/sadptprj_riclyap_adi"
export PYTHONPATH="$PYTHONPATH:$MYPYPATH/dolfin_navier_scipy"
export PYTHONPATH="$PYTHONPATH:$MYHOMEPATH/work/code/lqgbt-oseen"

RE=40
NTS=20000
PROBLEM=cylinderwake
MSHLVL=1

# RE=60
# NTS=20000

INIPERTURB=0.0
TRUNCAT=.01
FBTYPE=-1  # no feedback
FBTYPE=2  # full state feedback
FBTYPE=1  # lqg-bt feedback
# FBTYPE=1  # lqg-bt feedback
PYMESS=1
FBTYPE=4  # hinf-bt feedback
NUMPICARDS=20

SCALETEST=.005

GRAMSPATH=/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/
HNFQR=_hinf.mat%outRegulator.Z%outFilter.Z%gam
LQGQR=_lqg.mat%Z_LQG_regulator%Z_LQG_filter

GRAMSFILE=${GRAMSPATH}cylinderwake_re${RE}${LQGQR}
GRAMSFILE=${GRAMSPATH}cylinderwake_re${RE}${HNFQR}

# LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

python3 compscript.py \
    --problem=${PROBLEM} --mesh=${MSHLVL} \
    --iniperturb=${INIPERTURB} --RE=${RE} \
    --closed_loop=${FBTYPE} --pymess \
    --scaletest=${SCALETEST} --truncat=${TRUNCAT} \
    --strtogramfacs=${GRAMSFILE} \
    --Nts=${NTS}
    # --ttf --ttf_npcrdstps=${NUMPICARDS}
