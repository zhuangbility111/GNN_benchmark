GROUP_ID=gac50544
output_dir=ogbn-products/log/check_breakdown/baseline/
# qsub -g $GROUP_ID -o ogbn-products/log/ -e ogbn-products/log/ ./run_ogbn-products_1.sh
qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_ogbn-products_2.sh
qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_ogbn-products_4.sh
# qsub -g $GROUP_ID -o ogbn-products/log/ -e ogbn-products/log/ ./run_ogbn-products_8.sh
# qsub -g $GROUP_ID -o ogbn-products/log/ -e ogbn-products/log/ ./run_ogbn-products_16.sh
# qsub -g $GROUP_ID -o ogbn-products/log/ -e ogbn-products/log/ ./run_ogbn-products_32.sh
# qsub -g $GROUP_ID -o ogbn-products/log/ -e ogbn-products/log/ ./run_ogbn-products_64.sh
# qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_ogbn-products_128.sh
qsub -g $GROUP_ID -o $output_dir -e $output_dir ./run_ogbn-products_256.sh
