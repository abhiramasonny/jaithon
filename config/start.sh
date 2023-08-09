#!/bin/bash

make

if [ "$#" -eq 0 ]; then
    ./interpreter
else
    ./interpreter $1
fi
