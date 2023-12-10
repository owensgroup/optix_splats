import os, sys

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)

import torch
torch_root = os.path.dirname(torch.__file__)
optix_root = '/usr/local/NVIDIA-OptiX-SDK-7.6.0-linux64-x86_64'
setup(
    name="diff_gaussian_renderer",
    version="1.0.0",
    description="Differentiable gaussian renderer using OptiX",
    license="MIT",
    packages=['diff_gaussian_renderer'],
    cmake_args=[f'-DCMAKE_PREFIX_PATH={torch_root}', f'-DOPTIX_HOME={optix_root}'])