import os

dirs = ["session11_","session12_","session13_"]#, "session2__", "session3__", "session4__", "session5__", "session6__", "session7__", "session8__"]

for indirname in dirs:
	outdirname = indirname + '__'
	try:
		os.mkdir(outdirname)
	except:
		pass

	for filename in os.listdir(indirname):
		fr = open(indirname + "/" + filename, "r")
		fw = open(outdirname + "/" + filename, "w")

		while True:
			line = fr.readline()
			if not line:
				break
			arr = line.split()
			if len(arr) != 3:
				continue
			try:
				for i in range((len(arr) - 1)):			
					tmp = float(arr[i])
				for i in range((len(arr) - 1)):
					fw.write(arr[i] + " ")
				fw.write(arr[len(arr) - 1] + "\n")
			except:
				pass
		fr.close()
		fw.close()

