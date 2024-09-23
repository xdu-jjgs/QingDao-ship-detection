from setuptools import setup, find_packages
from distutils.command.build_ext import build_ext as _build_ext
from Cython.Build import cythonize
import os
import shutil


def find_all_pys(dir:str):
    '''多级目录下找到所有.py文件
    '''
    py_files = []
    for dirpath, _, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(".py"):
                py_files.append(os.path.join(dirpath, filename))
    return py_files


def find_all_files(path_list:list[str]):
    '''获取项目中所有待编译的.py文件
    '''
    all_py_files = []
    for path in path_list:
        if os.path.isdir(path):
            all_py_files += find_all_pys(path)
        elif os.path.isfile(path):
            all_py_files += [path]
        else:
            print(f"{path} 不是有效的文件或文件夹路径")
    return all_py_files


def compile_all_modules(build_dir:str, all_py_files:list[str]):
    '''编译每个文件，跳过出错的文件
    '''
    successful_modules = []
    failed_modules = []

    for py_file in all_py_files:
        try:
            # 单个文件进行 cythonize
            successful_modules.extend(cythonize(
                module_list=py_file,
                # 设置 .c 文件的构建目录
                build_dir=build_dir,  
                compiler_directives={'language_level': "3"}
            ))
        except Exception as e:
            print(f"Error compiling {py_file}, skipping. Error: {e}")
            failed_modules.append(py_file)
    # 打印跳过的文件
    if failed_modules:
        print(f"The following files were skipped due to errors: {failed_modules}")
    return successful_modules


def arrange_compiled_files(src_dir:str, tgt_dir:str):
    '''将编译的文件移动到新目录
    '''
    if not os.path.isdir(tgt_dir):os.makedirs(tgt_dir)
    for dirpath, dirnames, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.endswith('.so') or filename.endswith('.pyd'):
                # 获取文件的完整路径
                src_file_path = os.path.join(dirpath, filename)
                # 计算相对于 src_dir 的相对路径
                relative_path = os.path.relpath(src_file_path, src_dir)
                # 在 tgt_dir 中创建相同的多级目录
                tgt_file_dir = os.path.join(tgt_dir, os.path.dirname(relative_path))
                os.makedirs(tgt_file_dir, exist_ok=True)
                # 目标文件路径
                tgt_file_path = os.path.join(tgt_file_dir, filename)
                # 移动文件
                shutil.move(src_file_path, tgt_file_path)
                print(f"Moved: {src_file_path} -> {tgt_file_path}")

if __name__ == '__main__':
    # 编译的c文件保存路径(可以删去)
    build_dir = './build_c_dir'
    # 项目根目录
    arrange_src_dir = './'
    # 编译后二进制文件保存路径(打包项目所在路径)
    arrange_tgt_dir = './build_pyd'
    # 列举出所有待编译路径
    compile_path = [
        # 'infer',
        # 'tracker',
        # 'ppocr',
        'constant.py',
        'keep_detect.py',
        'model.py',
        'run.py',
        'utils.py',
        'ws_handler.py'
    ]
    all_py_files = find_all_files(compile_path)
    # 编译每个文件，跳过出错的文件
    successful_modules = compile_all_modules(build_dir, all_py_files)
    # 打包
    setup(
        name='ship_infer',
        version='0.1',
        packages=find_packages(),
        ext_modules=successful_modules,
        cmdclass={'build_ext': _build_ext},
    )
    arrange_compiled_files(arrange_src_dir, arrange_tgt_dir)
