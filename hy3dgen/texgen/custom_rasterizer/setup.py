from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import subprocess
import torch

def is_rocm_available():
    """Check if ROCm is available by checking for hipcc and PyTorch ROCm support."""
    # Check if hipcc is in PATH
    try:
        result = subprocess.run(
            ['hipcc', '--version'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Detected ROCm (hipcc available)")
            return True
    except FileNotFoundError:
        pass

    # Check if PyTorch is built with ROCm support
    try:
        import torch
        if torch.version.hip is not None:
            print("Detected PyTorch with ROCm support")
            return True
    except (ImportError, AttributeError):
        pass

    print("Using CUDA (nvcc)")
    return False

def get_cpp_config_paths():
    """Find the correct c++config.h path for hipcc."""
    cpp_config_paths = []
    for path in [
        '/usr/include/c++/15/x86_64-redhat-linux',
        '/usr/include/c++/14/x86_64-redhat-linux',
        '/usr/include/c++/13/x86_64-redhat-linux',
        '/usr/include/c++/12/x86_64-redhat-linux',
        '/usr/include/c++/11/x86_64-redhat-linux',
    ]:
        if os.path.exists(os.path.join(path, 'bits', 'c++config.h')):
            cpp_config_paths.append(path)
            break
    return cpp_config_paths


def build_extension():
    """Build the custom rasterizer extension with appropriate backend."""
    use_rocm = is_rocm_available()

    if use_rocm:
        # ROCm/HIP backend
        extra_include_paths = get_cpp_config_paths()
        ext_module = CUDAExtension(
            'custom_rasterizer_kernel',
            [
                'lib/custom_rasterizer_kernel/rasterizer_hip.cpp',
                'lib/custom_rasterizer_kernel/grid_neighbor_hip.cpp',
                'lib/custom_rasterizer_kernel/rasterizer_gpu.hip',
            ],
            extra_compile_args={
                'cxx': ['-O2', '-D__AMDGCN_WAVEFRONT_SIZE=32'],
                'nvcc': ['-O2', '-D__AMDGCN_WAVEFRONT_SIZE=32']
            },
            include_dirs=extra_include_paths
        )
        print("Building with HIP (ROCm) backend")
    else:
        # CUDA backend
        ext_module = CUDAExtension(
            'custom_rasterizer_kernel',
            [
                'lib/custom_rasterizer_kernel/rasterizer.cpp',
                'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
                'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
            ]
        )
        print("Building with CUDA (nvcc) backend")

    return [ext_module]




torch_version = torch.__version__.split('+')[0].replace('.', '')
if torch.version.cuda is not None:
    cuda_version = "cuda"+torch.version.cuda.replace('.', '')
elif getattr(torch.version, 'hip', None) is not None:
    cuda_version = "rocm" + torch.version.hip.replace('.','')

version = f"0.1.0+torch{torch_version}.{cuda_version}"
# build custom rasterizer
# build with `python setup.py install`
# nvcc/hipcc is needed

custom_rasterizer_module = CUDAExtension('custom_rasterizer_kernel', [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
    'lib/custom_rasterizer_kernel/grid_neighbor.cpp',
    'lib/custom_rasterizer_kernel/rasterizer_gpu.cu',
])

setup(
    packages=find_packages(),
    version=version,
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=build_extension(),
    cmdclass={
        'build_ext': BuildExtension
    },   
)
