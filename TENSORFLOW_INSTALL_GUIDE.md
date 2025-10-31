# TensorFlow Installation Guide for Windows

## Problem
TensorFlow installation is failing due to Windows Long Path limitation (260 character limit).

## Solution: Enable Long Path Support

### Option 1: Using PowerShell Script (Recommended)

1. **Open PowerShell as Administrator:**
   - Press `Win + X`
   - Select "Windows PowerShell (Admin)" or "Terminal (Admin)"

2. **Navigate to the project directory:**
   ```powershell
   cd D:\DS_project
   ```

3. **Allow script execution (if needed):**
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. **Run the script:**
   ```powershell
   .\enable_long_paths.ps1
   ```

5. **Restart your computer** (important!)

6. **After restart, install TensorFlow:**
   ```powershell
   cd D:\DS_project
   .\.venv\Scripts\Activate
   pip install tensorflow --no-cache-dir
   ```

---

### Option 2: Manual Registry Edit

1. **Open Registry Editor:**
   - Press `Win + R`
   - Type `regedit` and press Enter
   - Click "Yes" to allow changes

2. **Navigate to:**
   ```
   HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem
   ```

3. **Create or modify:**
   - Right-click in the right pane
   - Select "New" → "DWORD (32-bit) Value"
   - Name it: `LongPathsEnabled`
   - Double-click and set Value to: `1`
   - Click OK

4. **Restart your computer**

5. **Install TensorFlow:**
   ```powershell
   pip install tensorflow --no-cache-dir
   ```

---

### Option 3: Use Group Policy (Windows 10 Pro/Enterprise)

1. **Open Group Policy Editor:**
   - Press `Win + R`
   - Type `gpedit.msc` and press Enter

2. **Navigate to:**
   ```
   Computer Configuration → Administrative Templates → System → Filesystem
   ```

3. **Enable the policy:**
   - Double-click "Enable Win32 long paths"
   - Select "Enabled"
   - Click "Apply" then "OK"

4. **Restart your computer**

5. **Install TensorFlow:**
   ```powershell
   pip install tensorflow --no-cache-dir
   ```

---

## Alternative: Use a Lighter ML Framework

If you continue having issues with TensorFlow, consider using:

### TensorFlow Lite
```powershell
pip install tensorflow-lite
```

### ONNX Runtime (smaller, faster)
```powershell
pip install onnxruntime
# Convert your .h5 model to ONNX format
```

### PyTorch (alternative framework)
```powershell
pip install torch torchvision
# You'll need to convert your model
```

---

## Verify Installation

After installing TensorFlow, verify it works:

```powershell
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU available:', tf.config.list_physical_devices('GPU'))"
```

---

## Using the Web App

### With Demo Mode (Current - No TensorFlow needed)
```powershell
python app_demo.py
```
- Uses random predictions
- Shows how the web interface works
- No model required

### With Real Model (After TensorFlow installation)
```powershell
python app.py
```
- Uses your trained model
- Provides real predictions
- Requires `marine_plastic_classifier.h5` file

---

## Troubleshooting

### Issue: "Module tensorflow.python not found"
**Solution:** TensorFlow installation is incomplete. Follow the Long Path guide above.

### Issue: "Model file not found"
**Solution:** Train your model first:
```powershell
python main.py --data_dir data --epochs 15
```

### Issue: "Permission denied"
**Solution:** Run PowerShell as Administrator for system changes.

### Issue: Installation still fails after enabling long paths
**Solutions:**
1. Make sure you restarted your computer
2. Try using a virtual environment in a shorter path:
   ```powershell
   cd C:\
   mkdir ml
   cd ml
   python -m venv venv
   .\venv\Scripts\Activate
   pip install tensorflow --no-cache-dir
   ```
3. Use WSL (Windows Subsystem for Linux) instead

---

## Quick Start Guide

1. **Enable Long Paths** (one-time setup)
2. **Restart computer**
3. **Install TensorFlow:**
   ```powershell
   pip install tensorflow --no-cache-dir
   ```
4. **Run the web app:**
   ```powershell
   python app.py
   ```
5. **Open browser:** http://localhost:5000

---

## Need Help?

- TensorFlow Documentation: https://www.tensorflow.org/install
- Windows Long Path Guide: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation
- Project Issues: Check README.md for common solutions
