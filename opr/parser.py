#! /usr/bin/env python

import csv
from subprocess import call
import sys

if (len(sys.argv[1:]) != 2):
	print "provide 2 arguments"
	print "usage: " + sys.argv[0], "fileToParse fileWithConstants"

else:
	csvfileName = sys.argv[2]
	lpfileName = sys.argv[1]

	csvfile = open(csvfileName, "rb")
	reader = csv.reader(csvfile, delimiter=' ')

	defines = {}

	for row in reader:
		if (len(row) == 2):
			defines[row[0]] = row[1]

	csvfile.close()

	lpfile = open(lpfileName, "rb")
	output = open("out.lp", "wb")

	for line in lpfile:
		for item in defines:
			line = line.replace(item, defines[item])
		output.write(line)

	output.close()
	lpfile.close()

	call(["lp_solve", "out.lp"])
