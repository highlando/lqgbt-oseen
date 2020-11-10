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
PYMESS=1
NUMPICARDS=6
SCALETEST=.3
NTS=25000
GRAMSFILE='/scratch/owncloud-gwdg/mpi-projects/18-hinf-lqgbt/results/cylinderwake_re50_hinf.mat%outRegulator.Z%outFilter.Z%gam'

LOGFILE=logs/N${CYLDIM}re${RE}fbt${FBTYPE}pm${PYMESS}nps${NUMPICARDS}trnc${TRUNCAT}sspd${INIPERTURB}st${SCALETEST}

echo tail -f $LOGFILE
source addpypath.sh
python3 compscript.py --iniperturb=${INIPERTURB} --RE=${RE} --RE_ini=${INIRE} \
    --closed_loop=${FBTYPE}  --ttf_npcrdstps=${NUMPICARDS} --pymess=${PYMESS} \
    --scaletest=${SCALETEST} --truncat=${TRUNCAT} --strtogramfacs=${GRAMSFILE} \
    --Nts=${NTS}
# --cyldim=${CYLDIM}  # >> $LOGFILE
