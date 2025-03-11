from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="my_cuda_extension",
    ext_modules=[
        CUDAExtension(
            name="my_cuda_extension",
            sources=["my_cuda_extension.cpp", "my_cuda_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)