GROUP_ID=gac50544
qsub -g $GROUP_ID ./run_1.sh
qsub -g $GROUP_ID ./run_2.sh
qsub -g $GROUP_ID ./run_4.sh
qsub -g $GROUP_ID ./run_8.sh
qsub -g $GROUP_ID ./run_16.sh
qsub -g $GROUP_ID ./run_32.sh
qsub -g $GROUP_ID ./run_64.sh
qsub -g $GROUP_ID ./run_128.sh
