from setuptools import setup, find_packages
from torch.utils import cpp_extension

compile_extra_args = ['-O3', '-Kopenmp', '-Nlibomp', '-Kfast', '-Kzfill']
link_extra_args = ['-Kopenmp', '-Nlibomp', '-Kfast', '-Kzfill']
# compile_extra_args = ['-O3', '-fopenmp', '-mavx512f', '-mavx512cd', '-mavx512bw', '-mavx512dq', '-mavx512vl', '-mavx512ifma', '-mavx512vbmi']
# link_extra_args = ['-fopenmp']

setup(name='quantization_cpu',
      ext_modules=[
          cpp_extension.CppExtension(
              'quantization_cpu',
              ['src/quantization.cpp'],
                extra_compile_args = compile_extra_args,
                extra_link_args = link_extra_args
          ),
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
    )
