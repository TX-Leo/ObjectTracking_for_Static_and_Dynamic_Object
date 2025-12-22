# Setup
```
conda create -n cotracker python=3.10
conda activate cotracker
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/co-tracker.git
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth -P checkpoints/
```

# Run
```
bash run_cotracker.sh
```