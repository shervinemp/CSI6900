#!/bin/bash
IFS='\n' VALUES=$(python3 ../mscripts/get_logs.py --grouped)
while read LINE; do
	#echo $LINE
	eval "python3 ../mscripts/time_to_csv.py $LINE"
done <<< "$VALUES"
