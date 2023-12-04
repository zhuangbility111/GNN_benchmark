import os
import re
import argparse
import pandas as pd


# label_list = ["fp32", "fp32 + pre_aggr", "random_quant", "random_quant + pre_aggr", "int2", "int2 + pre_aggr"]
label_list = {"ori": "fp32", "ori+pre": "fp32 + pre_aggr", "ori+int2": "int2", "ori+pre+int2": "fp32 + int2 + pre_aggr"}


def extract_numbers_from_line(line):
    numbers = re.findall(r"\d+\.\d+|\d+", line)
    if len(numbers) == 0:
        return None
    return numbers


def calculate_average(numbers):
    if numbers:
        return sum(numbers) / len(numbers)
    else:
        return 0


def sort_dict_by_key(dict):
    # sort the average_dict by key
    sorted_dict = sorted(dict.items(), key=lambda x: x[0])
    return sorted_dict


def collect_avg_epoch_time(sorted_statistic_dict, output_file):
    avg_epoch_time = dict()
    avg_epoch_time["world_size"] = list()
    
    for world_size, value in sorted_statistic_dict:
        # print("world_size: {}".format(world_size))
        avg_epoch_time["world_size"].append(world_size)
        for optimization_type, data in value.items():
            print(optimization_type)
            if label_list[optimization_type] not in avg_epoch_time:
                avg_epoch_time[label_list[optimization_type]] = list()
            # avg_epoch_time[label_list[optimization_type]].append(calculate_average(data["epoch_time"]))
            # don't include the first epoch
            avg_epoch_time[label_list[optimization_type]].append(calculate_average(data["epoch_time"][1:]))
    print("avg_epoch_time = {}".format(avg_epoch_time))
    df = pd.DataFrame(avg_epoch_time)
    if os.path.exists(output_file):
        os.remove(output_file)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="avg_epoch_time", index=False)


def collect_test_acc(sorted_statistic_dict, output_file):
    for world_size, value in sorted_statistic_dict:
        test_acc_dict = dict()
        test_acc_dict["epoch"] = list()
        num_epochs = 0
        for optimization_type, data in value.items():
            test_acc_dict[label_list[optimization_type]] = data["test_acc"]
            num_epochs = len(data["test_acc"])
        test_acc_dict["epoch"] = [i for i in range(num_epochs)]
        print("world_size: {}, num_epochs; {}".format(world_size, num_epochs))

        df = pd.DataFrame(test_acc_dict)
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a") as writer:
            df.to_excel(writer, sheet_name="test_acc_on_{}".format(world_size), index=False)


def collect_statistic_data(input_dir):
    perf_target_string = "Epoch:"  # 请将 "目标字符串" 替换为你要搜索的字符串

    # 获取当前目录下的所有文件
    level1_files = os.listdir(input_dir)
    # {world_size: {tpye: data list}}
    statistic_dict = dict()


    # level 1, n proc folder
    for level1_f in level1_files:
        # use re to extract the number as world_size
        if level1_f[-4:] != "proc":
            continue
        print(level1_f)
        world_size = re.findall(r"\d+", level1_f)[0]
        level2_files = os.listdir(os.path.join(input_dir, level1_f))
        level2_files.sort()
        print(level2_files)
        # level 2, type folder (fp32, fp32 + pre_aggr, ...)
        for level2_f in level2_files:
            level3_files = os.listdir(os.path.join(input_dir, level1_f, level2_f))
            # level 3, rank folder (rank0, rank1, ...)
            for level3_f in level3_files:
                # use re to extract the number as rank
                rank = int(re.findall(r"\d+", level3_f)[-1])
                if rank == 0 and level3_f[0:6] == 'stdout':
                    file = os.path.join(input_dir, level1_f, level2_f, level3_f)
                    print("Processing file: {}".format(file))
                    is_data_valid = False
                    with open(file, "r") as f:
                        lines = f.readlines()
                        # 遍历每一行
                        for line in lines:
                            # extract the performance numbers as the value
                            if perf_target_string in line:
                                # Rank: 0, World_size: 32, Epoch: 144, Loss: 0.4007447063922882, Train: 0.9194, Val: 0.9094, Test: 0.7783, Time: 1.884016
                                numbers = extract_numbers_from_line(line)
                                if numbers is not None:
                                    if len(numbers) == 8:
                                        rank = int(numbers[-8])
                                        world_size = int(numbers[-7])
                                        epoch = int(numbers[-6])
                                        loss = float(numbers[-5])
                                        train_acc = float(numbers[-4])
                                        val_acc = float(numbers[-3])
                                        test_acc = float(numbers[-2])
                                        epoch_time = float(numbers[-1])
                                    elif len(numbers) == 7:
                                        loss = 0.0
                                        rank = int(numbers[-7])
                                        world_size = int(numbers[-6])
                                        epoch = int(numbers[-5])
                                        train_acc = float(numbers[-4])
                                        val_acc = float(numbers[-3])
                                        test_acc = float(numbers[-2])
                                        epoch_time = float(numbers[-1])

                                    key = world_size
                                    if key not in statistic_dict:
                                        statistic_dict[key] = dict()
                                        
                                    if level2_f not in statistic_dict[key]:
                                        statistic_dict[key][level2_f] = dict()
                                        statistic_dict[key][level2_f]["loss"] = []
                                        statistic_dict[key][level2_f]["train_acc"] = []
                                        statistic_dict[key][level2_f]["val_acc"] = []
                                        statistic_dict[key][level2_f]["test_acc"] = []
                                        statistic_dict[key][level2_f]["epoch_time"] = []

                                    statistic_dict[key][level2_f]["loss"].append(loss)
                                    statistic_dict[key][level2_f]["train_acc"].append(train_acc)
                                    statistic_dict[key][level2_f]["val_acc"].append(val_acc)
                                    statistic_dict[key][level2_f]["test_acc"].append(test_acc)
                                    statistic_dict[key][level2_f]["epoch_time"].append(epoch_time)

                            # remove the \r\n at the end of the line
                            if line.strip() == "training end.":
                                is_data_valid = True

                    if is_data_valid == False:
                        world_size = int(world_size)
                        print("data in world_size {} is invalid.".format(world_size))
                        if world_size in statistic_dict.keys():
                            statistic_dict.pop(world_size)

    return statistic_dict


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--input_dir", "-i", type=str, default=".")
    args.add_argument("--output_dir", "-o", type=str, default=".")
    args.add_argument("--graph_name", "-g", type=str, default=".")

    args = args.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    graph_name = args.graph_name

    print("input_dir: {}".format(input_dir))
    print("output_dir: {}".format(output_dir))
    print("graph_name: {}".format(graph_name))

    statistic_dict = collect_statistic_data(input_dir)
    sorted_statistic_dict = sort_dict_by_key(statistic_dict)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "{}_result.xlsx".format(graph_name))
    collect_avg_epoch_time(sorted_statistic_dict, output_file)
    collect_test_acc(sorted_statistic_dict, output_file)
