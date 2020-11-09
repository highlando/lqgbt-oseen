# # need to specify
#  * there: RE 
#  * there: fb type
#  * inival: enforce ss+d via iniperturb
#  * npcrd
#  * there: truncat
#  * there: scaletest
#  * C
#  * logfile <--- gonna use shell redirect

# python3 compscript.py --iniperturb=0.0001 --re=100 --closed_loop=4

RE=50
INIRE=45
# CYLDIM=3
INIPERTURB=0.0
TRUNCAT=0.01
FBTYPE=4  # hinf-bt feedback
FBTYPE=2  # full state feedback
FBTYPE=-1  # no feedback
FBTYPE=1  # lqg-bt feedback
# FBTYPE=1  # lqg-bt feedback
NUMPICARDS=6
SCALETEST=.3
NTS=36000

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

echo tail -f $LOGFILE
source addpypath.sh
python3 compscript.py --iniperturb=${INIPERTURB} --RE=${RE} --RE_ini=${INIRE} \
    --closed_loop=${FBTYPE}  --ttf_npcrdstps=${NUMPICARDS} \
    --scaletest=${SCALETEST} --truncat=${TRUNCAT} \
    --Nts=${NTS}
# --cyldim=${CYLDIM}  # >> $LOGFILE
