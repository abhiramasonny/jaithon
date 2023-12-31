#!/bin/bash

directory="Test/Jaithon"

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
      ./jaithon "$directory_filename"
    fi
  fi
done
