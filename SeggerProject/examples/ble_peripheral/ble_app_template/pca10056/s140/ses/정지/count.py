
# # bgzfile = open("sit_ax.txt", 'r')
# # print(len(bgzfile.readlines()))
# # bgzfile.close()
# # bgzfile = open("sit_ay.txt", 'r')
# # print(len(bgzfile.readlines()))
# # bgzfile.close()
# # bgzfile = open("sit_az.txt", 'r')
# # print(len(bgzfile.readlines()))
# # bgzfile.close()
# # bgzfile = open("sit_gx.txt", 'r')
# # print(len(bgzfile.readlines()))
# # bgzfile.close()
# # bgzfile = open("sit_gy.txt", 'r')
# # print(len(bgzfile.readlines()))
# # bgzfile.close()
# # bgzfile = open("sit_gz.txt", 'r')
# # print(len(bgzfile.readlines()))
# # bgzfile.close()

# import random
# ylist = []
# bgxlist = []
# bgylist = []
# bgzlist = []
# taxlist = []
# taylist = []
# tazlist = []

# yfile = open("y_test_kjh5.txt", 'r')
# for y in yfile:
#     ylist.append(y)
# yfile.close()

# bgxfile = open("body_gyro_x_test_kjh5.txt", 'r')
# for bgx in bgxfile:
#     bgxlist.append(bgx)
# bgxfile.close()
# bgyfile = open("body_gyro_y_test_kjh5.txt", 'r')
# for bgy in bgyfile:
#     bgylist.append(bgy)
# bgyfile.close()
# bgzfile = open("body_gyro_z_test_kjh5.txt", 'r')
# for bgz in bgzfile:
#     bgzlist.append(bgz)
# bgzfile.close()

# taxfile = open("total_acc_x_test_kjh5.txt", 'r')
# for tax in taxfile:
#     taxlist.append(tax)
# taxfile.close()
# tayfile = open("total_acc_y_test_kjh5.txt", 'r')
# for tay in tayfile:
#     taylist.append(tay)
# tayfile.close()
# tazfile = open("total_acc_z_test_kjh5.txt", 'r')
# for taz in tazfile:
#     tazlist.append(taz)
# tazfile.close()

# yfile = open("y_test_kjh6.txt", 'w')
# bgxfile = open("body_gyro_x_test_kjh6.txt", 'w')
# bgyfile = open("body_gyro_y_test_kjh6.txt", 'w')
# bgzfile = open("body_gyro_z_test_kjh6.txt", 'w')
# taxfile = open("total_acc_x_test_kjh6.txt", 'w')
# tayfile = open("total_acc_y_test_kjh6.txt", 'w')
# tazfile = open("total_acc_z_test_kjh6.txt", 'w')

# shuffleList = []
# for i in range(45):
#     shuffleList.append(i)
# random.shuffle(shuffleList)

# for i in shuffleList:
#     yfile.write(ylist[i])
#     bgxfile.write(bgxlist[i])
#     bgyfile.write(bgylist[i])
#     bgzfile.write(bgzlist[i])
#     taxfile.write(taxlist[i])
#     tayfile.write(taylist[i])
#     tazfile.write(tazlist[i])
# yfile.close()
# bgxfile.close()
# bgyfile.close()
# bgzfile.close()
# taxfile.close()
# tayfile.close()
# tazfile.close()

bgzfile = open("body_gyro_z_train_kjh6.txt", 'r')
newbgzfile = open("body_gyro_z_train_kjh7.txt", 'w')

for line in bgzfile:
    bgzlist = line.strip().split(' ')
    bgzlist = bgzlist[0:128]
    for bgz in bgzlist:
        newbgzfile.write(' '+bgz)
    newbgzfile.write('\n')

bgzfile.close()
newbgzfile.close()