#!/bin/bash

RE=60
NTS=3072  # 12*2**8
PROBLEM=dbrotcyl
MSHLVL=2

SHRTLF=resultsoverview-dbrotcyl.md

FBTYPE=4  # hinf-bt feedback

REPTSL=(-192 -135 -96 -48 -24 -12 -6)
TRUNCATSL=(.512 .128 .064 .045 .032 .008)

SCALETEST=25.

GRAMSPATH=testdata/

HNFQR=_hinf.mat%outRegulator.Z%outFilter.Z%gam  # if mat73 is installed
HNFQR=_hinfv5.mat%outRegulator%outFilter%gam  # else
GRAMSFILE=${GRAMSPATH}doublecylinder_re${RE}${HNFQR}

SHRTLF=resultsoverview-2.md

for REPT in "${REPTSL[@]}"; do
    for THRSH in "${TRUNCATSL[@]}"; do
        python3 compscript.py \
            --problem=${PROBLEM} --mesh=${MSHLVL} \
            --RE=${RE} \
            --closed_loop=${FBTYPE} \
            --scaletest=${SCALETEST} --truncat=${THRSH} \
            --strtogramfacs=${GRAMSFILE} \
            --Nts=${NTS} \
            --ttf_ptre --ttf_value=${REPT} \
            --shortlogfile=${SHRTLF}
    done
done
