# %% Load Raw Data (Collected from NI DAQ)
def LoadSoftSensorData(SubjectNo = -1):
	import bz2
	import gzip
	import os
	import pickle
	import numpy as np

	# Default Data Path
	DataRootPath = "./Data"

	if (SubjectNo == -1):
		SubjectNos = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
		SubjectNames = ["S{:02d}".format(no) for no in SubjectNos]
	else:
		if (np.isscalar(SubjectNo)):
			SubjectNos = SubjectNo
			SubjectNames = ["S{:02d}".format(SubjectNo)]
		else:
			SubjectNos = SubjectNo
			SubjectNames = ["S{:02d}".format(no) for no in SubjectNos]

	# Group Name - "Untitled"
	# Channel Name
	# Time                  Time
	# Voltage_S0            Untitled
	# Voltage_S1            Untitled 1
	# Voltage_S2            Untitled 2
	# Voltage_S3            Untitled 3
	# Voltage_S4            Untitled 4
	# Voltage_S5            Untitled 5
	# Voltage_S6            Untitled 6
	# Voltage_S7            Untitled 7
	# Voltage_S8            Untitled 8
	# Voltage_S9            Untitled 9
	# DigitalIn_SW          Untitled 10


	AllData = {}
	print("Load Raw NI Data. ", end=" ")
	for subj in SubjectNames:
		fn = os.path.join(DataRootPath, '{}.pgz'.format(subj))
		#print("Load {} Data.".format(subj))
		print(subj, end=".")
		with gzip.GzipFile(fn, 'r') as f:
			AllData[subj] = pickle.load(f)
	print(" Done.")

	return (AllData, SubjectNames, SubjectNos)








# %% Load Segmented Data (MotionID, MotionName, Start Transient, Stable Region, End Transient)
def LoadSegData(SubjectNo = -1):
	import bz2
	import gzip
	import os
	import pickle
	import numpy as np

	# Default Data Path
	DataRootPath = "./Data/Seg"

	if (SubjectNo == -1):
		SubjectNos = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
		SubjectNames = ["S{:02d}".format(no) for no in SubjectNos]
	else:
		if (np.isscalar(SubjectNo)):
			SubjectNos = SubjectNo
			SubjectNames = ["S{:02d}".format(SubjectNo)]
		else:
			SubjectNos = SubjectNo
			SubjectNames = ["S{:02d}".format(no) for no in SubjectNos]

	AllData = {}
	print("Load Seg Data. ", end=" ")
	for subj in SubjectNames:
		fn = os.path.join(DataRootPath, '{}.seg'.format(subj))
		#print("Load {} Data.".format(subj))
		print(subj, end=".")
		with gzip.GzipFile(fn, 'r') as f:
			AllData.update(pickle.load(f))
	print(" Done.")

	return (AllData, SubjectNames, SubjectNos)


# %% Load Windowed Data (ex.{S00}-{200}_{50}-{Dyn,Sta,All})
def LoadWdwData(WindowSize, OverlapSize, Tag, SubjectNo = -1):
	import bz2
	import gzip
	import os
	import pickle
	import numpy as np

	# Default Data Path
	DataRootPath = "./Data/Window"

	if (SubjectNo == -1):
		SubjectNos = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
		SubjectNames = ["S{:02d}".format(no) for no in SubjectNos]
	else:
		if (np.isscalar(SubjectNo)):
			SubjectNos = SubjectNo
			SubjectNames = ["S{:02d}".format(SubjectNo)]
		else:
			SubjectNos = SubjectNo
			SubjectNames = ["S{:02d}".format(no) for no in SubjectNos]

	AllData = {}
	print("Load Wdw Data. ", end=" ")
	for subj in SubjectNames:
		fn = os.path.join(DataRootPath, '{}-{}.{}-{}.wdw'.format(subj, WindowSize, OverlapSize, Tag))
		#print("Load {} Data.".format(subj))
		print(subj, end=".")
		with gzip.GzipFile(fn, 'r') as f:
			SubjName, Wdw_Data, WindowSize_, OverlapSize_, Tag_ = pickle.load(f)
			assert(WindowSize == WindowSize_ and OverlapSize == OverlapSize_ and Tag.lower() == Tag_.lower())
			AllData.update({SubjName: Wdw_Data})
	print(" Done.")

	return (AllData, SubjectNames, SubjectNos)


#a = LoadWdwData([1, 2])
