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
INIPERTURB=0.0001
TRUNCAT=0.01
# FBTYPE=1
FBTYPE=4
PYMESS=1
NUMPICARDS=6
SCALETEST=1.

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

echo tail -f $LOGFILE
python3 compscript.py --obsperturb=1 --iniperturb=${INIPERTURB} --re=${RE} --closed_loop=${FBTYPE}  --ttf_npcrdstps=${NUMPICARDS} --pymess=${PYMESS} --scaletest=${SCALETEST} --truncat=${TRUNCAT} --cyldim=${CYLDIM} # >> $LOGFILE
