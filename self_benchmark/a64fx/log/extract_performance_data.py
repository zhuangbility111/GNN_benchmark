import re
import argparse

res = dict()
res['forward_time(ms)'] = list()
res['backward_time(ms)'] = list()
res['update_weight_time(ms)'] = list()
res['load_data_and_model_time(ms)'] = list()
res['total_training_time(ms)'] = list()

parser = argparse.ArgumentParser(description='input file information')
parser.add_argument("-i", "--input_file", dest='input_file', type=str, help='The path of input file.')

args = parser.parse_args()
input_file = args.input_file
output_file = './csv/' + input_file + '.csv'

with open(input_file) as f_read:
    for line in f_read:
        if line.find("time") != -1:
            tmp = line.split(',')
            res[tmp[0]].append(tmp[1].replace("\n", "").replace(":", ""))
# print(res)

with open(output_file, 'w') as f_write:
    for k, v in res.items():
        f_write.write(k + ',')
        for i in v:
            f_write.write(i + ',')
        f_write.write('\r\n')

print("over")
