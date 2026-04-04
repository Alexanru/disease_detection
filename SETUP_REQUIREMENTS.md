# ⚙️ Setup Requirements

What to install on your laptop to run RareSight.

---

## **Mandatory** (Everyone)

### **1. Python 3.11+**
- **Download:** [python.org/downloads](https://www.python.org/downloads/)
- **Install:** Windows → Run installer, check ✅ "Add Python to PATH"
- **Verify:**
  ```powershell
  python --version    # Should show 3.11+
  ```

### **2. Visual Studio Code**
- **Download:** [code.visualstudio.com](https://code.visualstudio.com/)
- **Install:** Standard installation
- **Extensions (recommended):**
  - Python (Microsoft)
  - Pylance
  - Thunder Client (for API testing)

### **3. Git**
- **Download:** [git-scm.com](https://git-scm.com/)
- **Install:** Standard, use default settings
- **Verify:**
  ```powershell
  git --version
  ```

---

## **Conditional** (Only if you have GPU)

### **For GTX 1650 (NVIDIA GPU)**

#### **CUDA 12.1**
- **Official:** [nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- **Steps:**
  1. Select: OS = Windows, Architecture = x86_64, Version = 12
  2. Installer type = exe (local)
  3. Download (~2.5 GB) → Run installer
  4. Keep defaults, install in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1`
- **Verify:**
  ```powershell
  nvcc --version    # Should show CUDA 12.1
  ```

#### **cuDNN 8.9+**
- **Official:** [nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
- **Steps:**
  1. Create free NVIDIA account (required)
  2. Download cuDNN 8.9.x for Windows (x86_64)
  3. Unzip to: `C:\Program Files\NVIDIA\cuDNN\bin`
  4. Add to PATH:
     ```powershell
     $env:PATH += ";C:\Program Files\NVIDIA\cuDNN\bin"
     # Make permanent:
     setx PATH "$env:PATH;C:\Program Files\NVIDIA\cuDNN\bin"
     ```
- **Verify:**
  ```powershell
  # Should find cuDNN DLL
  ls "C:\Program Files\NVIDIA\cuDNN\bin\cudnn*.dll"
  ```

##### **Common Errors:**
- **"CUDA not found"** → Reinstall CUDA, check PATH
- **"cuDNN not found"** → Check `C:\Program Files\NVIDIA\cuDNN\bin` exists

---

## **Recommended** (Quality of Life)

### **Docker** (for containerized training)
- **Download:** [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)
- **Install:** Standard
- **Verify:**
  ```powershell
  docker --version
  ```

### **Devbox** (for automated environment)
- **Download:** [jetify.com/devbox](https://www.jetify.com/devbox)
- **Install:** Follow setup wizard
- **Verify:**
  ```powershell
  devbox --version
  ```

---

## **Installation Checklist**

Go through this in order:

```
☐ Python 3.11+ installed and in PATH
☐ Visual Studio Code installed  
☐ Git installed
☐ (If GTX 1650) CUDA 12.1 installed
☐ (If GTX 1650) cuDNN in PATH
☐ (Optional) Docker installed
☐ (Optional) Devbox installed
```

---

## **Verify Installation**

Run this in PowerShell to check everything:

```powershell
Write-Host "Python:" $(python --version)
Write-Host "Git:" $(git --version)
Write-Host "CUDA:" $(nvcc --version 2>&1 | Select-String "release")
Write-Host "Docker:" $(docker --version 2>&1)
Write-Host "Devbox:" $(devbox --version 2>&1)
```

---

## **Next Steps**

Once everything is installed:
1. Clone or download RareSight repository
2. See [TRAINING_STEPS.md](TRAINING_STEPS.md) for training
3. See [DEVBOX_QUICK.md](DEVBOX_QUICK.md) if using Devbox

---

## **Troubleshooting**

### **Python not recognized**
- Symptom: `python: command not found`
- Fix: Add to PATH (reinstall with ✅ "Add Python to PATH")

### **CUDA not working on GTX 1650**
- Symptom: PyTorch says "CUDA not available"
- Fix:
  ```powershell
  # Update NVIDIA drivers first
  # Then run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

### **cuDNN DLL not found**
- Symptom: Runtime error about cuDNN
- Fix: 
  ```powershell
  # Check cuDNN location
  ls "C:\Program Files\NVIDIA\cuDNN\bin\cudnn*.dll"
  
  # Add to PATH permanently
  setx PATH "$env:PATH;C:\Program Files\NVIDIA\cuDNN\bin"
  
  # Restart VS Code
  ```

### **Out of Memory (OOM)**
- Symptom: "CUDA out of memory" during training
- Fix: Reduce `batch_size` in config files (try 8 or 4)

---

## **Questions?**

See [INDEX.md](INDEX.md) for more guides.
