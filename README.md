# Optix Splats
This repo contains code for a differentiable gaussian renderer that leverages ray-tracing and OIT
techniques to render gaussian splatted scenes. 

## Setup and Compilation
First install Anaconda. Then create an environment from the repo yaml
```
conda env create -f environment.yml
```
### Compilation
You can build the extension with
```
python setup.py install
```

You can test the install is functional by running the test script
```
python test/render_test.py
```
### Common Issues
You may have issues with pygame if on Ubuntu 22.04. To fix you need
to delete the conda versions of libstdc++ so it falls back to your OS.
You can find the path to the conda drivers by exposing the debug environment variable.
```
export LIBGL_DEBUG=verbose
```
And then going to the anaconda path and deleting libstdc++ (deleting the symlink works too)