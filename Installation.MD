✅ Step-by-step (clean & safe)
1️⃣ Go to your project folder
cd path\to\your\project

2️⃣ Create a virtual environment
python -m venv venv

3️⃣ Activate the venv
.\venv\Scripts\Activate.ps1


You should see:

(venv) PS C:\...

4️⃣ Upgrade pip (important)
python -m pip install --upgrade pip

5️⃣ Install CUDA-enabled PyTorch FIRST

⚠️ Do this before installing requirements.txt

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

6️⃣ Verify GPU works
python - <<EOF
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
EOF


Expected:

True
NVIDIA GeForce RTX 3060 Ti


If this is not correct → stop here and tell me.

7️⃣ Install the rest of your requirements

Now install everything else:

pip install -r requirements.txt