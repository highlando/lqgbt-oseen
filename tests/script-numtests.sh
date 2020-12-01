MYHOMEPATH=/home/heiland
MYPYPATH=/home/heiland/software/gits/mypys
export PYTHONPATH="$MYPYPATH/mat_lib_plots:$MYPYPATH/dolfin_navier_scipy"
export PYTHONPATH="$PYTHONPATH:$MYHOMEPATH/work/code/lqgbt-oseen"
export PYTHONPATH="$PYTHONPATH:$MYPYPATH/sadptprj_riclyap_adi"

RE=60
INIPERTURB=0.0
TRUNCAT=0.001
FBTYPE=4  # hinf-bt feedback
FBTYPE=2  # full state feedback
FBTYPE=-1  # no feedback
FBTYPE=1  # lqg-bt feedback
# FBTYPE=1  # lqg-bt feedback
NUMPICARDS=20
SCALETEST=1.5
NTS=15000

GRAMSPATH=/scratch/tbd/dnsdata/
GRAMSSPEC=cylinderwake_Re${RE}.0_gamma1.0_NV41718_Bbcc_C31_palpha1e-05__
LQGQR=%zwc.npy%zwo.npy

GRAMSFILE=${GRAMSPATH}${GRAMSSPEC}${LQGQR}

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

echo tail -f $LOGFILE
python3 compscript.py --iniperturb=${INIPERTURB} --RE=${RE} \
    --closed_loop=${FBTYPE} \
    --scaletest=${SCALETEST} --truncat=${TRUNCAT} \
    --strtogramfacs=${GRAMSFILE} \
    --Nts=${NTS} \
    --ttf --ttf_npcrdstps=${NUMPICARDS}
# --cyldim=${CYLDIM}  # >> $LOGFILE
