#!/bin/bash

TYPE=("loss" "gradient" "entropy" "uncertainty")
NBATCHES=(2 5 7)
CUMULATIVE=("True" "False")
FORWARD=("True" "False")
EPOCHS=(10 20 30)
CEPOCHS=(10 20 30)


for t in "${TYPE[@]}"
do
    for nb in "${NBATCHES[@]}"
    do
        for c in "${CUMULATIVE[@]}"
        do
            for f in "${FORWARD[@]}"
            do
                for e in "${EPOCHS[@]}"
                do
                    for ce in "${CEPOCHS[@]}"
                    do
                        python automated_curricula.py --curric_type=t --numbatches=nb --cumulative=c --forward=f --epochs=e --curriculum_epochs=ce --plot=False
                    done
                done
            done
        done
    done
done