from setuptools import setup, find_packages
from torch.utils import cpp_extension
import subprocess
import glob


def check_cpu_support(extra_compile_args, extra_link_args, source_files):
    try:
        result = subprocess.run(["lscpu"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if "avx512f" in result.stdout:
            extra_compile_args.extend(
                [
                    "-fopenmp",
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512ifma",
                    "-mavx512vbmi",
                ]
            )
            extra_link_args.extend(
                [
                    "-fopenmp",
                    "-mavx512f",
                    "-mavx512cd",
                    "-mavx512bw",
                    "-mavx512dq",
                    "-mavx512vl",
                    "-mavx512ifma",
                    "-mavx512vbmi",
                ]
            )
            source_files.extend(glob.glob("src/*_x86.cpp"))
        elif "sve" in result.stdout:
            extra_compile_args.extend(["-Kopenmp", "-Nlibomp", "-Kfast", "-Kzfill"])
            extra_link_args.extend(["-Kopenmp", "-Nlibomp", "-Kfast", "-Kzfill"])
            source_files.extend(glob.glob("src/*_arm.cpp"))
        else:
            print("Warning: CPU does not support AVX512 or SVE")

    except Exception as e:
        print(f"Error: {e}")


def get_extensions():
    extra_compile_args = ["-O3"]
    extra_link_args = []
    source_files = ["src/utils.cpp", "src/spmm.cpp", "src/ops.cpp"]
    check_cpu_support(extra_compile_args, extra_link_args, source_files)

    extensions = []
    extensions.append(
        cpp_extension.CppExtension(
            "computation_ops",
            source_files,
            include_dirs=["src/"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    )
    return extensions


setup(
    name="computation_ops",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    packages=find_packages(),
)
