MYHOMEPATH=/home/heiland
MYPYPATH=/home/heiland/software/gits/mypys
export PYTHONPATH="$MYPYPATH/mat_lib_plots:$MYPYPATH/dolfin_navier_scipy"
export PYTHONPATH="$PYTHONPATH:$MYHOMEPATH/work/code/lqgbt-oseen"

RE=60
INIRE=55
# CYLDIM=3
INIPERTURB=0.0
TRUNCAT=0.001
FBTYPE=4  # hinf-bt feedback
FBTYPE=2  # full state feedback
FBTYPE=-1  # no feedback
FBTYPE=1  # lqg-bt feedback
# FBTYPE=1  # lqg-bt feedback
NUMPICARDS=6
NUMPICARDS=-1
SCALETEST=1.
NTS=20000

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

echo tail -f $LOGFILE
python3 compscript.py --iniperturb=${INIPERTURB} --RE=${RE} --RE_ini=${INIRE} \
    --closed_loop=${FBTYPE}  --ttf_npcrdstps=${NUMPICARDS} \
    --scaletest=${SCALETEST} --truncat=${TRUNCAT} \
    --Nts=${NTS}
# --cyldim=${CYLDIM}  # >> $LOGFILE
