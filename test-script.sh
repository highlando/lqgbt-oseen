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
INIPERTURB=0.
TRUNCAT=0.05
FBTYPE=4
FBTYPE=-1
# 2: full_output_fb, 1: red_output_fb, 4: hinf_red_output_fb, -1: no_feedback
NUMPICARDS=3
SCALETEST=1.5

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}
echo tail -f $LOGFILE
export PYTHONPATH=''
python3 compscript.py --obsperturb=1 --iniperturb=${INIPERTURB} --re=${RE} --closed_loop=${FBTYPE}  --ttf_npcrdstps=${NUMPICARDS} --scaletest=${SCALETEST} --truncat=${TRUNCAT} --cyldim=${CYLDIM} >> $LOGFILE
