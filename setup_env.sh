module load cuda12.2/
module load mamba
mamba init
source /burg/home/yy3448/.bashrc
# module load cuda
# mamba init
# source ~/.bashrc

if mamba env list | grep -q 'peak_traj'; then
    echo "Environment 'peak_traj' found. Activating it..."
    mamba activate peak_traj
else
    echo "Environment 'peak_traj' not found. Creating it..."
    mamba create -n peak_traj python=3.11 --yes
    mamba activate peak_traj
fi

pip install pandas numpy==1.26 anndata muon scikit-learn umap-learn
pip install pyranges gffutils pybedtools
pip install scipy h5py tables mofapy2 flit
pip install gtfparse intervaltree igraph leidenalg
pip install 'gene-trajectory[widgets]'

mamba install ipykernel ipywidgets scanpy biotite matplotlib seaborn --yes