import csv
import os
import sys
from datetime import datetime, timedelta

import glob

import argparse
parser 	= argparse.ArgumentParser(description='')
parser.add_argument('--csv_path', default=None, type=str, help='')
parser.add_argument('--datetime', default=None, type=str, help='')
args 	= parser.parse_args()

def generate_amv_bst(filepath):
	with open(filepath, 'r') as bstfile:
		reader = csv.reader(bstfile, delimiter=',')
		index = 0
		for row in reader:
			print("Generating AMV for TC:", row[7])
			bt_ID = int(row[1])
			numOfDataLines = int(row[2])
			for i in range(numOfDataLines):
				line = next(reader)
				
				tc_type = int(line[2])

				datetime_str = line[0]
				yyyy = 2000 + int(datetime_str[0:2])
				mm = (int)(datetime_str[2:4])
				dd = (int)(datetime_str[4:6])
				hh = (int)(datetime_str[6:8])

				anchor_time = datetime(yyyy, mm, dd, hh, 0, 0)
				prev_time = anchor_time - timedelta(minutes=10)
				next_time = anchor_time + timedelta(minutes=10)

				anchor_name = "{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}".format(anchor_time.year,
																			anchor_time.month,
																			anchor_time.day, 
																			anchor_time.hour, 
																			anchor_time.minute)
				prev_name 	= "{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}".format(prev_time.year,
																			prev_time.month, 
																			prev_time.day, 
																			prev_time.hour, 
																			prev_time.minute)
				next_name 	= "{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}".format(next_time.year,
																			next_time.month, 
																			next_time.day, 
																			next_time.hour, 
																			next_time.minute)

				if 	os.path.isfile(os.path.join(RAW_PPM_DIR, "%s.ppm" % anchor_name)) 	and \
					os.path.isfile(os.path.join(RAW_PPM_DIR, "%s.ppm" % prev_name)) 	and \
					os.path.isfile(os.path.join(RAW_PPM_DIR, "%s.ppm" % next_name)):
					# calculate amv
					os.system("bm 15 4 2 7 %s %s.ppm %s.ppm %s %s 0" % (RAW_PPM_DIR, prev_name, anchor_name, AMV_PPM_DIR, anchor_name))
					os.system("bm 15 4 2 7 %s %s.ppm %s.ppm %s %s 0" % (RAW_PPM_DIR, anchor_name, next_name, AMV_PPM_DIR, "_%s" % anchor_name))

def generate_amv_datetime(datetime_str):
	img_ppm_dir 	= "..\\..\\data\\raw\\tc\\doksuri\\img_ppm\\"
	amv_ppm_dir 	= "..\\..\\data\\raw\\tc\\doksuri\\amv_ppm\\"

	yyyy 			= 2000 + int(datetime_str[0:2])
	mm 				= (int)(datetime_str[2:4])
	dd 				= (int)(datetime_str[4:6])
	hh 				= (int)(datetime_str[6:8])
	current_time 	= datetime(yyyy, mm, dd, hh, 0, 0)
	for prev in range(1, 25, 1):
		anchor_time 	= current_time 	- timedelta(hours=prev)
		prev_time 		= anchor_time 	- timedelta(minutes=10)
		next_time 		= anchor_time 	+ timedelta(minutes=10)

		anchor_name 	= "{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}".format(anchor_time.year,
																	anchor_time.month,
																	anchor_time.day, 
																	anchor_time.hour, 
																	anchor_time.minute)
		prev_name 		= "{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}".format(prev_time.year,
																	prev_time.month, 
																	prev_time.day, 
																	prev_time.hour, 
																	prev_time.minute)
		next_name 		= "{:0>4d}{:0>2d}{:0>2d}{:0>2d}{:0>2d}".format(next_time.year,
																	next_time.month, 
																	next_time.day, 
																	next_time.hour, 
																	next_time.minute)

		if 	os.path.isfile(os.path.join(img_ppm_dir, "%s.ppm" % anchor_name)) and \
			os.path.isfile(os.path.join(img_ppm_dir, "%s.ppm" % prev_name)) and \
			os.path.isfile(os.path.join(img_ppm_dir, "%s.ppm" % next_name)):
			# calculate amv
			os.system("bm 15 4 2 7 %s %s.ppm %s.ppm %s %s 0" % (img_ppm_dir, prev_name, anchor_name, amv_ppm_dir, anchor_name))
			os.system("bm 15 4 2 7 %s %s.ppm %s.ppm %s %s 0" % (img_ppm_dir, anchor_name, next_name, amv_ppm_dir, "_%s" % anchor_name))

if __name__ == '__main__':
	if args.csv_path is not None:
		generate_amv_bst(args.csv_path)
	elif args.datetime is not None:
		generate_amv_datetime(args.datetime)