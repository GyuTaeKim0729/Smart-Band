import os

indirname = "session21"
outdirname = "session21_"

os.mkdir(outdirname)
for filename in os.listdir(indirname):
	fr = open(indirname + "/" + filename, "r")
	fw = open(outdirname + "/" + filename.split(".")[0].split("_")[1] + ".log", "w")

	while True:
		line = fr.readline()
		if not line:
			break
		arr = line.split()
		if len(arr) < 3:
			continue
		for i in range(0, (len(arr) - 1)):
			fw.write(arr[i] + " ")
		fw.write(arr[len(arr) - 1] + "\n")

	fr.close()
	fw.close()
