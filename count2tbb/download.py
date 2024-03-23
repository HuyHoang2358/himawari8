import argparse

import csv
import os
import sys
from datetime import datetime, timedelta

parser 	= argparse.ArgumentParser(description='')
parser.add_argument('--csv_path', default=None, type=str, help='')
parser.add_argument('--datetime', default=None, type=str, help='')
args 	= parser.parse_args()

def download_bst(filepath):
	with open(filepath, 'rb') as bstfile:
		reader = csv.reader(bstfile, delimiter=',')
		index = 0
		for row in reader:
			print "Downloading TC:", row[7]
			bt_ID = int(row[1])
			numOfDataLines = int(row[2])
			for i in range(numOfDataLines):
				line = reader.next()
				
				tc_type = int(line[2])

				datetime_str = line[0]
				yyyy = 2000 + int(datetime_str[0:2])
				mm = (int)(datetime_str[2:4])
				dd = (int)(datetime_str[4:6])
				hh = (int)(datetime_str[6:8])

				anchor_time = datetime(yyyy, mm, dd, hh, 0, 0)
				for delta in [-10, 0, 10]:
					target_time = anchor_time - timedelta(minutes=delta)
					if not os.path.isfile("../../data/tropical-cyclone/raw/{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld.tif".format(target_time.year,
																																	target_time.month, 
																																	target_time.day, 
																																	target_time.hour, 
																																	target_time.minute)):
						os.system("./count2tbb.sh {:0>4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}".format(target_time.year,
																							target_time.month, 
																							target_time.day, 
																							target_time.hour, 
																							target_time.minute))
					
def download_datetime(datetime_str):
	yyyy 		= 2000 + int(datetime_str[0:2])
	mm 			= (int)(datetime_str[2:4])
	dd 			= (int)(datetime_str[4:6])
	hh 			= (int)(datetime_str[6:8])
	current_time 	= datetime(yyyy, mm, dd, hh, 0, 0)
	for prev in range(1, 25, 1):
		anchor_time 	= current_time - timedelta(hours=prev)
		for delta in [-10, 0, 10]:
			target_time = anchor_time - timedelta(minutes=delta)
			if not os.path.isfile("../../../data/raw/tc/doksuri/img/{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}.tir.01.fld.tif".format(target_time.year,
																															target_time.month, 
																															target_time.day, 
																															target_time.hour, 
																															target_time.minute)):
				os.system("./count2tbb.sh {:0>4d} {:0>2d} {:0>2d} {:0>2d} {:0>2d}".format(target_time.year,
																					target_time.month, 
																					target_time.day, 
																					target_time.hour, 
																					target_time.minute))

if __name__ == '__main__':
	if args.csv_path is not None:
		download_bst(args.csv_path)
	elif args.datetime is not None:
		download_datetime(args.datetime)