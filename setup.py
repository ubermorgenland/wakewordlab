import subprocess
import sys
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext as _build_ext
import numpy as np

try:
    from Cython.Build import cythonize
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False
    print("WARNING: Cython not found — extension will not be compiled.")

extra_compile_args = [
    "-O2",
    "-fvisibility=hidden",   # hide all symbols not explicitly exported
    "-fstack-protector",
]
extra_link_args = []

if sys.platform.startswith("linux"):
    extra_compile_args.append("-fstack-protector-strong")
    extra_link_args.append("-Wl,--strip-all")  # strip symbols at link time


class build_ext(_build_ext):
    """Post-build: strip debug symbols from the compiled extension."""
    def run(self):
        super().run()
        if sys.platform == "darwin":
            for ext in self.extensions:
                so = Path(self.get_ext_fullpath(ext.name))
                if so.exists():
                    subprocess.run(["strip", "-x", str(so)], check=False)


if HAS_CYTHON:
    extensions = cythonize(
        [
            Extension(
                "wakewordlab._loader.loader",
                sources=[
                    "wakewordlab/_loader/loader.pyx",
                    "wakewordlab/_loader/key.c",
                ],
                include_dirs=[
                    np.get_include(),
                    "wakewordlab/_loader",
                ],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link_args,
            )
        ],
        compiler_directives={"language_level": "3"},
    )
else:
    extensions = []

setup(
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={"build_ext": build_ext},
)
