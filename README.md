Needed package:

!pip install torch-scatter torch-cluster

!pip install torch-geometric

!pip install ase e3nn
 
The main Notebook is EGNN.ipynb

<img width="929" alt="Screenshot 2024-10-27 at 11 08 42" src="https://github.com/user-attachments/assets/70753beb-3894-40d3-b713-f3d597a810b7">


## Installation with GPU

```bash
pip install torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
CUDA=cu121
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+${CUDA}.html
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.4.0+${CUDA}.html

pip install torch-geometric
pip install ase e3nn scikit-learn matplotlib jupyterlab pandas plotly
```
