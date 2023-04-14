import numpy as np

part = 512
comm = np.load('global_comm_{}.npy'.format(part))
comm_volume = np.zeros((part, part), dtype=np.int64)

for i in range(part):
    for j in range(part):
        comm_volume[i][j] = comm[i][j] + comm[j][i]

num_nodes = part
num_edges = np.count_nonzero(comm_volume) / 2

with open('global_comm_{}.part'.format(part), 'w') as file:
    file.write("{} {} 001\n".format(num_nodes, int(num_edges)))
    for i in range(part):
        for j in range(part):
            if comm_volume[i][j] != 0:
                file.write(str(j+1))
                file.write(" ")
                file.write(str(comm_volume[i][j]))
                file.write(" ")
        file.write("\n")
