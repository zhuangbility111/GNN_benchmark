import os
import re
import argparse
import pandas as pd


# label_list = ["fp32", "fp32 + pre_aggr", "random_quant", "random_quant + pre_aggr", "int2", "int2 + pre_aggr"]
label_list = ["fp32", "fp32 + pre_aggr", "int2", "int2 + pre_aggr"]


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


def print_loss_and_acc(sorted_statistic_dict, end_epoch, label):
    print_loss_buf = ""
    print_train_acc_buf = ""
    print_val_acc_buf = ""
    print_test_acc_buf = ""

    print(label)
    for world_size, value in sorted_statistic_dict:
        # print("world_size: {}".format(world_size))
        print_loss_buf += "{}, ".format(value["loss"][end_epoch - 1])
        print_train_acc_buf += "{}, ".format(value["train_acc"][end_epoch - 1])
        print_val_acc_buf += "{}, ".format(value["val_acc"][end_epoch - 1])
        print_test_acc_buf += "{}, ".format(value["test_acc"][end_epoch - 1])

    # print("loss: [{}]".format(print_loss_buf))
    print("train_acc: [{}]".format(print_train_acc_buf))
    print("val_acc: [{}]".format(print_val_acc_buf))
    print("test_acc: [{}]".format(print_test_acc_buf))


def print_epoch_time(sorted_statistic_dict, begin_epoch, end_epoch, label):
    print_epoch_time_buf = ""

    # print(label)
    for world_size, value in sorted_statistic_dict:
        # print("world_size: {}".format(world_size))
        print_epoch_time_buf += "{:.6f}, ".format(
            calculate_average(value["epoch_time"][begin_epoch:end_epoch])
        )

    print("epoch_time: [{}]".format(print_epoch_time_buf))


def collect_avg_epoch_time(sorted_statistic_dict, num_epochs, output_file):
    avg_epoch_time = dict()
    avg_epoch_time["world_size"] = list()
    for i in range(len(label_list)):
        avg_epoch_time[label_list[i]] = list()
    for world_size, value in sorted_statistic_dict:
        # print("world_size: {}".format(world_size))
        avg_epoch_time["world_size"].append(world_size)
        for i in range(len(label_list)):
            avg_epoch_time[label_list[i]].append(
                calculate_average(value["epoch_time"][i * num_epochs + 1: (i + 1) * num_epochs])
            )
    df = pd.DataFrame(avg_epoch_time)
    if os.path.exists(output_file):
        os.remove(output_file)
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="avg_epoch_time", index=False)


def collect_test_acc(sorted_statistic_dict, num_epochs, output_file):
    for world_size, value in sorted_statistic_dict:
        test_acc_dict = dict()
        test_acc_dict["epoch"] = [i for i in range(num_epochs)]
        for i in range(len(label_list)):
            test_acc_dict[label_list[i]] = value["test_acc"][i * num_epochs : (i + 1) * num_epochs]
        df = pd.DataFrame(test_acc_dict)
        with pd.ExcelWriter(output_file, engine="openpyxl", mode="a") as writer:
            df.to_excel(writer, sheet_name="test_acc_on_{}".format(world_size), index=False)


def collect_statistic_data(input_dir):
    perf_target_string = "Epoch:"  # 请将 "目标字符串" 替换为你要搜索的字符串

    print("input_dir: {}".format(input_dir))
    # 获取当前目录下的所有文件
    files = os.listdir(input_dir)
    statistic_dict = dict()

    # 遍历每个文件
    for file in files:
        file = os.path.join(input_dir, file)
        if os.path.isfile(file):
            print("Processing file: {}".format(file))
            with open(file, "r") as f:
                lines = f.readlines()
                # 遍历每一行
                for line in lines:
                    # extract the performance numbers as the value
                    if perf_target_string in line:
                        # Rank: 0, World_size: 32, Epoch: 144, Loss: 0.4007447063922882, Train: 0.9194, Val: 0.9094, Test: 0.7783, Time: 1.884016
                        numbers = extract_numbers_from_line(line)
                        if numbers is not None:
                            rank = int(numbers[-8])
                            world_size = int(numbers[-7])
                            epoch = int(numbers[-6])
                            loss = float(numbers[-5])
                            train_acc = float(numbers[-4])
                            val_acc = float(numbers[-3])
                            test_acc = float(numbers[-2])
                            epoch_time = float(numbers[-1])

                            key = world_size
                            if key not in statistic_dict:
                                statistic_dict[key] = dict()
                                statistic_dict[key]["loss"] = []
                                statistic_dict[key]["train_acc"] = []
                                statistic_dict[key]["val_acc"] = []
                                statistic_dict[key]["test_acc"] = []
                                statistic_dict[key]["epoch_time"] = []

                            statistic_dict[key]["loss"].append(loss)
                            statistic_dict[key]["train_acc"].append(train_acc)
                            statistic_dict[key]["val_acc"].append(val_acc)
                            statistic_dict[key]["test_acc"].append(test_acc)
                            statistic_dict[key]["epoch_time"].append(epoch_time)

    return statistic_dict


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--num_epochs", "-e", type=int, default=250)
    args.add_argument("--input_dir", "-i", type=str, default=".")
    args.add_argument("--output_dir", "-o", type=str, default=".")
    args.add_argument("--graph_name", "-g", type=str, default=".")

    args = args.parse_args()
    num_epochs = args.num_epochs
    input_dir = args.input_dir
    output_dir = args.output_dir
    graph_name = args.graph_name

    print("num_epochs: {}".format(num_epochs))
    print("input_dir: {}".format(input_dir))
    print("output_dir: {}".format(output_dir))
    print("graph_name: {}".format(graph_name))

    statistic_dict = collect_statistic_data(input_dir)
    sorted_statistic_dict = sort_dict_by_key(statistic_dict)

    # for i in range(len(label_list)):
    #     print_loss_and_acc(sorted_statistic_dict, (i + 1) * num_epochs, label_list[i])
    #     print_epoch_time(sorted_statistic_dict, i * num_epochs, (i + 1) * num_epochs, label_list[i])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "{}_result.xlsx".format(graph_name))
    collect_avg_epoch_time(sorted_statistic_dict, num_epochs, output_file)
    collect_test_acc(sorted_statistic_dict, num_epochs, output_file)

    # print_loss_and_acc(sorted_statistic_dict, num_epochs, "original version:")
    # print_epoch_time(sorted_statistic_dict, 0, num_epochs, "original version:")

    # print_loss_and_acc(sorted_statistic_dict, 2 * num_epochs, "only pre_delay_aggr version:")
    # print_epoch_time(sorted_statistic_dict, num_epochs, 2 * num_epochs, "only pre_delay_aggr version:")

    # print_loss_and_acc(sorted_statistic_dict, 3 * num_epochs, "only random_quant version:")
    # print_epoch_time(sorted_statistic_dict, 2 * num_epochs, 3 * num_epochs, "only random_quant version:")

    # print_loss_and_acc(sorted_statistic_dict, 4 * num_epochs, "pre_delay_aggr + random_quant version:")
    # print_epoch_time(
    #     sorted_statistic_dict, 3 * num_epochs, 4 * num_epochs, "pre_delay_aggr + random_quant version:"
    # )

# collect_epoch_time(statistic_dict)
# collect_loss_and_acc(statistic_dict)

# ori_average_dict = {}
# only_pre_average_dict = {}
# only_quant_average_dict = {}
# pre_quant_average_dict = {}
# num_epochs = 250
# for key, numbers in perf_number_dict.items():
#     if len(numbers) > 1:
#         ori_average_dict[key] = calculate_average(numbers[1:num_epochs])
#         only_pre_average_dict[key] = calculate_average(numbers[num_epochs + 1 : 2 * num_epochs])
#         only_quant_average_dict[key] = calculate_average(numbers[2 * num_epochs + 1 : 3 * num_epochs])
#         pre_quant_average_dict[key] = calculate_average(numbers[3 * num_epochs + 1 :])

# for key, numbers in perf_number_dict.items():
#     print("Key: {}, Numbers: {}".format(key, numbers))

# for key, average in ori_average_dict.items():
#     print("Key: {}, origin Average: {}".format(key, average))

# for key, average in only_pre_average_dict.items():
#     print("Key: {}, only pre_delay_aggr Average: {}".format(key, average))

# for key, average in only_quant_average_dict.items():
#     print("Key: {}, only random_quant Average: {}".format(key, average))

# for key, average in pre_quant_average_dict.items():
#     print("Key: {}, pre_delay_aggr + random_quant Average: {}".format(key, average))

# print_sorted_dict(ori_average_dict, "original version:")
# print_sorted_dict(only_pre_average_dict, "only pre_delay_aggr version:")
# print_sorted_dict(only_quant_average_dict, "only random_quant version:")
# print_sorted_dict(pre_quant_average_dict, "pre_delay_aggr + random_quant version:")
