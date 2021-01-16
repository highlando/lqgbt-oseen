MYHOMEPATH=/home/heiland
MYPYPATH=/home/heiland/software/mypys
# export PYTHONPATH="$MYPYPATH/mat_lib_plots:$MYPYPATH/sadptprj_riclyap_adi"
export PYTHONPATH="$MYHOMEPATH/work/code/lqgbt-oseen"
export PYTHONPATH="$PYTHONPATH:$MYPYPATH/dolfin_navier_scipy"
echo $PYTHONPATH

RE=60
NTS=160000
PROBLEM=cylinderwake
MSHLVL=1
SHRTLF=resultsoverview.md

# RE=60
# NTS=20000

INIPERTURB=0.0
TRUNCAT=.00015
TRUNCAT=.15
FBTYPE=2  # full state feedback
FBTYPE=1  # lqg-bt feedback
# FBTYPE=1  # lqg-bt feedback
PYMESS=1
FBTYPE=-1  # no feedback
FBTYPE=4  # hinf-bt feedback
NUMPICARDS=15

SCALETEST=1.

GRAMSPATH=/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/
HNFQR=_hinf.mat%outRegulator.Z%outFilter.Z%gam
LQGQR=_lqg.mat%Z_LQG_regulator%Z_LQG_filter

GRAMSFILE=${GRAMSPATH}cylinderwake_re${RE}${LQGQR}
GRAMSFILE=${GRAMSPATH}cylinderwake_re${RE}${HNFQR}

python3 compscript.py \
    --problem=${PROBLEM} --mesh=${MSHLVL} \
    --iniperturb=${INIPERTURB} --RE=${RE} \
    --closed_loop=${FBTYPE} --pymess \
    --scaletest=${SCALETEST} --truncat=${TRUNCAT} \
    --strtogramfacs=${GRAMSFILE} \
    --Nts=${NTS} \
    --ttf_ssit --ttf_value=${NUMPICARDS} \
    --shortlogfile=${SHRTLF}
