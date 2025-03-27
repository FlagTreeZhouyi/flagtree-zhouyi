[English](./README.md)

## FlagTree

FlagTree 是多后端的 Triton 编译器项目。FlagTree 致力于打造多元 AI 芯片编译器及相关工具平台，发展和壮大 Triton 上下游生态。项目当前处于初期，目标是兼容现有适配方案，统一代码仓库，快速实现单版本多后端支持。

## Install from source
安装依赖（注意使用正确的 python3.x 执行）：
```shell
apt install zlib1g zlib1g-dev libxml2 libxml2-dev  # ubuntu
cd python; python3 -m pip install -r requirements.txt
```

编译安装，目前支持的后端 backendxxx 包括 iluvatar、xpu、mthreads、cambricon（有限支持）：
```shell
cd python
export FLAGTREE_BACKEND=backendxxx
python3 -m pip install . --no-build-isolation -v
```

## Tips for building

使用默认的编译命令，可以编译安装 nvidia、amd、triton_shared 后端：
```shell
# 自行下载 llvm
cd ${YOUR_LLVM_DOWNLOAD_DIR}
wget https://oaitriton.blob.core.windows.net/public/llvm-builds/llvm-10dc3a8e-ubuntu-x64.tar.gz
tar -zxvf llvm-10dc3a8e-ubuntu-x64.tar.gz
# 编译安装
cd ${YOUR_CODE_DIR}/flagtree/python
export LLVM_BUILD_DIR=${YOUR_LLVM_DOWNLOAD_DIR}/llvm-10dc3a8e-ubuntu-x64
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
unset FLAGTREE_BACKEND
python3 -m pip install . --no-build-isolation -v
# 如果接下来需要编译安装其他后端，应清空 LLVM 相关环境变量
unset LLVM_BUILD_DIR LLVM_INCLUDE_DIRS LLVM_LIBRARY_DIR LLVM_SYSPATH
```

## Running tests

安装完成后可以在后端目录下运行测试：
```shell
cd third_party/backendxxx/python/test
python3 -m pytest -s
```

## Contributing

欢迎参与 FlagTree 的开发并贡献代码，详情请参考[CONTRIBUTING.md](/CONTRIBUTING_cn.md)。

## License

FlagTree 使用 [MIT license](/LICENSE)。
