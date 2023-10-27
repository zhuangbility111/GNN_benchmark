GROUP_ID=gac50544
output_dir=ogbn-papers100M/log/check_breakdown/optimized/
# qsub -g $GROUP_ID -o ogbn-papers100M/log/ -e ogbn-papers100M/log/ ./run_ogbn-papers100M_2.sh
# qsub -g $GROUP_ID -o ogbn-papers100M/log/ -e ogbn-papers100M/log/ ./run_ogbn-papers100M_4.sh
# qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_ogbn-papers100M_8.sh
# qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_ogbn-papers100M_16.sh
# qsub -g $GROUP_ID -o ogbn-papers100M/log/ -e ogbn-papers100M/log/ ./run_ogbn-papers100M_32.sh
# qsub -g $GROUP_ID -o ogbn-papers100M/log/ -e ogbn-papers100M/log/ ./run_ogbn-papers100M_64.sh
# qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_ogbn-papers100M_128.sh
qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_ogbn-papers100M_256.sh
