GROUP_ID=gac50544
output_dir=reddit/log/check_breakdown/optimized/
# qsub -g $GROUP_ID -o reddit/log/ -e reddit/log/ ./run_reddit_1.sh
qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_reddit_2.sh
qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_reddit_4.sh
qsub -g $GROUP_ID -o reddit/log/ -e reddit/log/ ./run_reddit_8.sh
qsub -g $GROUP_ID -o reddit/log/ -e reddit/log/ ./run_reddit_16.sh
qsub -g $GROUP_ID -o reddit/log/ -e reddit/log/ ./run_reddit_32.sh
qsub -g $GROUP_ID -o reddit/log/ -e reddit/log/ ./run_reddit_64.sh
qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_reddit_128.sh
qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_reddit_256.sh
