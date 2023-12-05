GROUP_ID=gac50544
# qsub -g $GROUP_ID -o uk-2007-02/log/ -e uk-2007-02/log/ ./run_uk-2007-02_2.sh
# qsub -g $GROUP_ID -o uk-2007-02/log/ -e uk-2007-02/log/ ./run_uk-2007-02_4.sh
# qsub -g $GROUP_ID -o uk-2007-02/log/ -e uk-2007-02/log/ ./run_uk-2007-02_8.sh
# qsub -g $GROUP_ID -o uk-2007-02/log/ -e uk-2007-02/log/ ./run_uk-2007-02_16.sh
# qsub -g $GROUP_ID -o uk-2007-02/log/ -e uk-2007-02/log/ ./run_uk-2007-02_32.sh
qsub -g $GROUP_ID -o uk-2007-02/log/ -e uk-2007-02/log/ ./run_uk-2007-02_64.sh
# qsub -g $GROUP_ID -o uk-2007-02/log/ -e uk-2007-02/log/ ./run_uk-2007-02_128.sh
# qsub -g $GROUP_ID -o uk-2007-02/log/ -e uk-2007-02/log/ ./run_uk-2007-02_256.sh
