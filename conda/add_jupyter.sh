envname=環境名
conda activate $envname
conda install ipykernel
python -m ipykernel install --user --name $envname
