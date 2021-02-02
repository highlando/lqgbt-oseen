#!/bin/bash

RE=60
NTS=16000
PROBLEM=cylinderwake
MSHLVL=1

FBTYPE=4  # hinf-bt feedback

TRUNCATSL=(.0001 .0004 .0016 .0032 .0064)
NUMPICARDSL=(17 24 34 37 40 48)

SCALETEST=2.5

GRAMSPATH=testdata/
HNFQR=_hinf.mat%outRegulator.Z%outFilter.Z%gam
LQGQR=_lqg.mat%Z_LQG_regulator%Z_LQG_filter

GRAMSFILE=${GRAMSPATH}cylinderwake_re${RE}${LQGQR}
GRAMSFILE=${GRAMSPATH}cylinderwake_re${RE}${HNFQR}
SHRTLF=resultsoverview-2-1.md

for NPCS in "${NUMPICARDSL[@]}"; do
    for THRSH in "${TRUNCATSL[@]}"; do
        python3 compscript.py \
            --problem=${PROBLEM} --mesh=${MSHLVL} --RE=${RE} \
            --closed_loop=${FBTYPE} \
            --scaletest=${SCALETEST} --truncat=${THRSH} \
            --strtogramfacs=${GRAMSFILE} \
            --Nts=${NTS} \
            --ttf_ssit --ttf_value=${NPCS} \
            --shortlogfile=${SHRTLF}
    done
done
