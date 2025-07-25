# ComfyUI Wrapper for [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2)

*Cutting-edge 3D textured model generation integrated seamlessly into ComfyUI*

---

![ComfyUI-Hunyuan3D-2](https://github.com/user-attachments/assets/a5fcd4e1-f9d1-4c21-b299-c9af85dee163)

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/downloads/)

---

## 🚀 Features

* Supports **Hunyuan3D-2** model for advanced 3D textured generation
* Automatic download and wrapping of diffusers models
* Highly optimized **Preview3D-node** for interactive model preview
* Optional high-performance rasterizer and mesh processor extensions
* Integration of **BPT** for improved shape generation (optional)

---

## 📂 Project Structure

```
ComfyUI-Hunyuan3DWrapper/
│
├─ ComfyUI/                          # ComfyUI base folder
│   └─ custom_nodes/
│       └─ ComfyUI-Hunyuan3DWrapper/  # This wrapper
│           ├─ hy3dgen/
│           │   ├─ texgen/
│           │   │   ├─ custom_rasterizer/      # Rasterizer source and build scripts
│           │   │   └─ differentiable_renderer/ # Mesh processor extension
│           │   └─ shapegen/
│           │       └─ bpt/                     # Optional BPT feature  
│           ├─ requirements.txt
│           ├─ wheels/                         # Precompiled wheels for rasterizer
│           └─ README.md
│
├─ models/                              # Place main .ckpt/.safetensors models here
│   └─ hunyuan3d-dit-v2-0.safetensors
│
└─ scripts/                            # Useful helper scripts (setup, upgrade, etc)
    ├─ install_dependencies.bat        # Windows installation helper
    ├─ compile_rasterizer.bat          # Compile rasterizer helper
    └─ upgrade_xatlas.bat              # Automated xatlas upgrade script
```

---

## ⚙️ Installation

### 1. Install dependencies

In your Python environment or portable ComfyUI:

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper/requirements.txt
```

Or with portable ComfyUI embedded python:

```bash
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Hunyuan3DWrapper\requirements.txt
```

### 2. Install precompiled rasterizer wheel (recommended for speed)

For Python 3.12 + torch 2.6.0 + CUDA 12.6 (Windows 11):

```bash
python_embeded\python.exe -m pip install ComfyUI\custom_nodes\ComfyUI-Hunyuan3DWrapper\wheels\custom_rasterizer-0.1.0+torch260.cuda126-cp312-cp312-win_amd64.whl
```

### 3. If precompiled wheel not suitable — compile yourself:

```bash
cd ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper/hy3dgen/texgen/custom_rasterizer
python setup.py install
```

Or for portable:

```bash
cd C:\ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-Hunyuan3DWrapper\hy3dgen\texgen\custom_rasterizer
C:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install .
```

---

### 4. Mesh Processor Extension (optional for vertex inpainting)

```bash
cd hy3dgen/texgen/differentiable_renderer
python_embeded\python.exe setup.py build_ext --inplace
```

---

### 5. Optional: Install BPT shape generator

```bash
pip install -r ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper/hy3dgen/shapegen/bpt/requirements.txt
```

Download weights:
`bpt-8-16-500m.pt` from
[Huggingface BPT weights](https://huggingface.co/whaohan/bpt/blob/refs%2Fpr%2F1/bpt-8-16-500m.pt)

Place weights in:
`ComfyUI/custom_nodes/ComfyUI-Hunyuan3DWrapper/hy3dgen/shapegen/bpt`

---

## 🛠 Xatlas Upgrade Procedure (Fix UV Wrapping on High-Poly Meshes)

Run from the **ComfyUI portable root folder**:

```bash
python_embeded\python.exe -m pip uninstall xatlas
git clone --recursive https://github.com/mworchel/xatlas-python.git
cd xatlas-python\extern
rmdir /s /q xatlas
git clone --recursive https://github.com/jpcy/xatlas
```

**Modify source for patch**
Edit `xatlas-python\extern\xatlas\source\xatlas\xatlas.cpp`:

* Line 6774: change `#if 0` to `//#if 0`
* Line 6778: change `#endif` to `//#endif`

Finally, install locally:

```bash
cd ..\..\..
python_embeded\python.exe -m pip install .
```

---

## 🔧 Handy Scripts (Windows batch examples)

### install\_dependencies.bat

```bat
@echo off
echo Installing dependencies...
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-Hunyuan3DWrapper\requirements.txt
pause
```

### compile\_rasterizer.bat

```bat
@echo off
cd ComfyUI\custom_nodes\ComfyUI-Hunyuan3DWrapper\hy3dgen\texgen\custom_rasterizer
python_embeded\python.exe setup.py install
pause
```

### upgrade\_xatlas.bat

```bat
@echo off
python_embeded\python.exe -m pip uninstall -y xatlas
git clone --recursive https://github.com/mworchel/xatlas-python.git
cd xatlas-python\extern
rmdir /s /q xatlas
git clone --recursive https://github.com/jpcy/xatlas
powershell -Command "(Get-Content .\xatlas\source\xatlas\xatlas.cpp) -replace '#if 0', '//#if 0' | Set-Content .\xatlas\source\xatlas\xatlas.cpp"
powershell -Command "(Get-Content .\xatlas\source\xatlas\xatlas.cpp) -replace '#endif', '//#endif' | Set-Content .\xatlas\source\xatlas\xatlas.cpp"
cd ..\..\..
python_embeded\python.exe -m pip install .
pause
```

---

## 📖 Usage Tips & Encouragement

* Make sure you have a **modern ComfyUI version** supporting Preview3D nodes.
* Keep your Python and CUDA versions consistent with your wheels/builds.
* When in doubt, compiling from source ensures full compatibility and future-proofing.
* Your dedication to pushing boundaries with Hunyuan3D will bring you closer to seamless 3D model creation workflows. Keep experimenting and innovating!
