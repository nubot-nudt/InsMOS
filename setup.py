import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension




def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


def write_version_to_file(version, target_file):
    with open(target_file, 'w') as f:
        print('__version__ = "%s"' % version, file=f)


if __name__ == '__main__':
    version = '0.1.0' 

    setup(
        name='insmos',
        version=version,
        description='InsMOS is a method for segmenting moving objects from point cloud',
        install_requires=[
            'numpy==1.18.1',
            'torch>=1.10',
            'easydict',
            'pyyaml'
        ],
        author='Neng Wang',
        packages=find_packages(exclude=['logs', 'docs', 'output']),
        cmdclass={'build_ext': BuildExtension},
        ext_modules=[
            make_cuda_ext(
                name='iou3d_nms_cuda',
                module='models.bbox_post_process',
                sources=[
                    'src/iou3d_cpu.cpp',
                    'src/iou3d_nms_api.cpp',
                    'src/iou3d_nms.cpp',
                    'src/iou3d_nms_kernel.cu',
                ]
            ),
            make_cuda_ext(
                name='Array_Index',
                module='models.utils',
                sources=[
                    'src/Array_Index.cpp',
                ]
            ),
        ],
    )
