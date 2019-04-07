# # need to specify
#  * there: RE 
#  * there: fb type
#  * inival: enforce ss+d via iniperturb
#  * npcrd
#  * there: truncat
#  * there: scaletest
#  * cyldim
#  * C
#  * logfile <--- gonna use shell redirect

# python3 compscript.py --iniperturb=0.0001 --re=100 --closed_loop=4

RE=100
CYLDIM=3
INIPERTURB=0.0
TRUNCAT=0.01
# FBTYPE=-1  # no feedback
# FBTYPE=1  # lqg-bt feedback
# FBTYPE=1  # lqg-bt feedback
FBTYPE=4  # hinf-bt feedback
PYMESS=1
NUMPICARDS=6
SCALETEST=2.5

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

echo tail -f $LOGFILE
source addpypath.sh
python3 compscript.py --iniperturb=${INIPERTURB} --re=${RE} --closed_loop=${FBTYPE}  --ttf_npcrdstps=${NUMPICARDS} --pymess=${PYMESS} --scaletest=${SCALETEST} --truncat=${TRUNCAT} --cyldim=${CYLDIM}  # >> $LOGFILE
