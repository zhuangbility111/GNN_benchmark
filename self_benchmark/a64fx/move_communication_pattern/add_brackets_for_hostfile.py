import os

num_nodes = 32
num_parts = 8
with open('global_comm_{}.part.part.{}'.format(num_nodes, num_parts)) as in_file:
    in_lines = in_file.readlines()
    with open('hostfile_{}.txt'.format(num_nodes), 'w') as out_file:
        for in_line in in_lines:
            idx = int(in_line)
            out_file.write("({})\n".format(idx))
        
