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

RE=90
CYLDIM=3
INIPERTURB=0.001
TRUNCAT=0.001
FBTYPE=4
NUMPICARDS=3
SCALETEST=1.

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}
echo tail -f $LOGFILE
python3 compscript.py --obsperturb=1 --iniperturb=${INIPERTURB} --re=${RE} --closed_loop=${FBTYPE}  --ttf_npcrdstps=${NUMPICARDS} --scaletest=${SCALETEST} --truncat=${TRUNCAT} --cyldim=${CYLDIM} >> $LOGFILE
