Guide to Setup on https://www.reddit.com/r/ROCm/comments/1nrsbdn/amd_gpus_with_flashattention_sageattention_on_wsl2/

ComfyUI Setup Guide for AMD GPUs with FlashAttention + SageAttention on WSL2
Reference: Original Japanese guide by kemari

Platform: Windows 11 + WSL2 (Ubuntu 24.04 - Noble) + RX 7900XTX

1. System Update and Python Environment Setup
Since this Ubuntu instance is dedicated to ComfyUI, I'm proceeding with root privileges.

Note: 'myvenv' is an arbitrary name - feel free to name it whatever you like

sudo su
apt-get update
apt-get -y dist-upgrade
apt install python3.12-venv

python3 -m venv myvenv
source myvenv/bin/activate
python -m pip install --upgrade pip
2. AMD GPU Driver and ROCm Installation
wget https://repo.radeon.com/amdgpu-install/6.4.4/ubuntu/noble/amdgpu-install_6.4.60404-1_all.deb
sudo apt install ./amdgpu-install_6.4.60404-1_all.deb
wget https://repo.radeon.com/amdgpu/6.4.4/ubuntu/pool/main/h/hsa-runtime-rocr4wsl-amdgpu/hsa-runtime-rocr4wsl-amdgpu_25.10-2209220.24.04_amd64.deb
sudo apt install ./hsa-runtime-rocr4wsl-amdgpu_25.10-2209220.24.04_amd64.deb
amdgpu-install -y --usecase=wsl,rocm --no-dkms

rocminfo
3. PyTorch ROCm Version Installation
pip3 uninstall torch torchaudio torchvision pytorch-triton-rocm -y

wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4/pytorch_triton_rocm-3.4.0%2Brocm6.4.4.gitf9e5bf54-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4/torch-2.8.0%2Brocm6.4.4.gitc1404424-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4/torchaudio-2.8.0%2Brocm6.4.4.git6e1c7fe9-cp312-cp312-linux_x86_64.whl
wget https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.4/torchvision-0.23.0%2Brocm6.4.4.git824e8c87-cp312-cp312-linux_x86_64.whl
pip install pytorch_triton_rocm-3.4.0+rocm6.4.4.gitf9e5bf54-cp312-cp312-linux_x86_64.whl torch-2.8.0+rocm6.4.4.gitc1404424-cp312-cp312-linux_x86_64.whl torchaudio-2.8.0+rocm6.4.4.git6e1c7fe9-cp312-cp312-linux_x86_64.whl torchvision-0.23.0+rocm6.4.4.git824e8c87-cp312-cp312-linux_x86_64.whl
4. Resolve Library Conflicts
location=$(pip show torch | grep Location | awk -F ": " '{print $2}')
cd ${location}/torch/lib/
rm libhsa-runtime64.so*
5. Clear Cache (if previously used)
rm -rf /home/username/.triton/cache
Replace 'username' with your actual username

6. Install FlashAttention + SageAttention
cd /home/username
git clone https://github.com/ROCm/flash-attention.git
cd flash-attention
git checkout main_perf
pip install packaging
FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" python setup.py install
pip install sageattention
7. File Replacements
Grant full permissions to subdirectories before replacing files:

chmod -R 777 /home/username
Flash Attention File Replacement
Replace the following file in myvenv/lib/python3.12/site-packages/flash_attn/utils/:

distributed.py

SageAttention File Replacements
Replace the following files in myvenv/lib/python3.12/site-packages/sageattention/:

attn_qk_int8_per_block.py

attn_qk_int8_per_block_causal.py

quant_per_block.py