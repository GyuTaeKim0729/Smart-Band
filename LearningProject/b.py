fr = open("6.log", "r")
fw = open("6_.log", "w")

while True:
	line = fr.readline()
	if not line:
		break
	arr = line.split()
	print(arr)
	if len(arr) < 11:
		continue
	for i in range(2, len(arr) - 1):
		fw.write(arr[i] + " ")
	fw.write(arr[len(arr) - 1] + "\n")

fr.close()
fw.close()
