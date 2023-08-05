#!/bin/bash

directory="Test"

for full_filename in "$directory"/*
do
  if [ -f "$full_filename" ]
  then
    filename=$(basename -- "$full_filename")
    extension="${filename##*.}"
    filename="${filename%.*}"

    if [ "$extension" == "jai" ]
    then
      directory_filename="$directory/$filename"

      echo "$filename"
      ./start.sh "$directory_filename"
    fi
  fi
done
