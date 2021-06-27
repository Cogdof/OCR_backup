import collections
from PIL import Image
import os

# read file, save average True / False case


log_path = '/home/mll/v_mll3/OCR_data/deep-text-recognition-benchmark-master/result/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth/log/'

file_list = os.listdir(log_path)

for i in file_list:
    txt_file_list = open(log_path+i, 'r')
    adv = open(log_path +"adv_"+i, 'w')
    local_true_aver = 0
    true_count = 0
    local_false_aver = 0
    false_count = 0
    print(i)
    for line in txt_file_list:
        adv.write(line)

        if '\t' in line:
            #print(line)
            line = line.replace(" ","")
            if line.split('\t')[2] == "True":
                local_true_aver  = local_true_aver + float(line.split("\t")[3])
                true_count = true_count+1
            else :
                local_false_aver = local_false_aver + float(line.split("\t")[3])
                false_count = false_count+1


    print("local_true_aver : ",local_true_aver/true_count)
    print("local_false_aver : ", local_false_aver / false_count)

    adv.write("\n")
    adv.write("local_true_aver :{} \n".format(local_true_aver / true_count))
    adv.write("local_false_aver :{} \n".format(local_false_aver / false_count))
    adv.write("total : {}, true :{}, false : {}".format(true_count+false_count ,true_count , false_count))

    txt_file_list.close()
    adv.close()