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

RE=60
CYLDIM=2
INIPERTURB=0.0
TRUNCAT=0.01
FBTYPE=-1
# FBTYPE=3
PYMESS=0
NUMPICARDS=-1
SCALETEST=.25

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

echo tail -f $LOGFILE
source addpypath.sh
python3 compscript.py --iniperturb=${INIPERTURB} --re=${RE} --closed_loop=${FBTYPE}  --ttf_npcrdstps=${NUMPICARDS} --pymess=${PYMESS} --scaletest=${SCALETEST} --truncat=${TRUNCAT} --cyldim=${CYLDIM}  # >> $LOGFILE
