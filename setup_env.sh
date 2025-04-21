module load cuda12.2/
module load mamba
mamba init
source /burg/home/yy3448/.bashrc

if mamba env list | grep -q 'peak'; then
    echo "Environment 'peak' found. Activating it..."
    mamba activate peak
else
    echo "Environment 'peak' not found. Creating it..."
    mamba create -n peak
    mamba activate peak
fi

pip install scanpy pandas numpy
mamba install ipykernel ipywidgets scanpy biotite matplotlib seaborn --yes