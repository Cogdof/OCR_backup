import os
import shutil

from PIL import Image
'''
#-----------------------------------------
MJ dataset gt generate
#------------------------------------------
'''
path = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/data/[IC15]ch4_training/'                  #기존 파일이 저장되어 있는 디렉토리

file_list = os.listdir(path)


                    #텍스트 파일 열기

gt_file2 = open(path + "/gt2.txt", 'w')
gt_file = open(path+"/gt.txt", 'r')
line = gt_file.readlines()
for i in line:
    i = i.replace(" ","\t")
    print(i)
    gt_file2.write("{}".format(i))




gt_file.close()
gt_file2.close()


'''

path = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/data'                  #기존 파일이 저장되어 있는 디렉토리

file_list = os.listdir(path)


for i in file_list:                         #텍스트 파일 열기


    gt_file = open(path+"/{}/gt.txt".format(i), 'a')
    img_file_list = os.listdir(path+"/{}/data/".format(i))
    for j in img_file_list:
        arr = j.split("_")
        label = arr[1]
        gt_file.write("data/{}\t{}\n".format(j, label))




    gt_file.close()

'''