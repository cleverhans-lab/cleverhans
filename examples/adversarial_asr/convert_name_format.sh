#!/bin/bash
for entry in "$HOME/tf/librispeech/raw/LibriSpeech/test-clean"/*
do 
  for i in "$entry"/*
  do     
    for j in "$i"/*.flac
    do 
      flac -d $j 
    done
  done
done
