## GPU 环境

运行环境统一为：

hendrixfut01fl

## 测试环境
准备工具：Nsight Compute

Nsight Compute在KU开发机安装步骤：

1. 官网下载Nsight Compute
https://developer.nvidia.com/nsight-compute

2. scp或者vscode拖到KU开发机

```shell
cd $HOME
mkdir -p =$HOME/my-tools/nsight_compute
sh ./nsight-compute-linux-*.run --noexec --target $HOME/my-tools/nsight_compute
readlink -f my-tools/nsight_compute//pkg/ncu #verification of the path of ncu
echo 'export PATH=$HOME/my-tools/nsight_compute/pkg:$PATH' >> ~/.bashrc
module load cuda
nvcc -o matmul_app matmul.cu -lineinfo
ncu ./matmul_app # 验证是不是撞上了ncu
```


## 测试

### Latency
latency统一测试参数：

num_attention_heads=16

hidden_size=512

hidden_dim=512/16=32

seqlen=1024

batch_size=1


跑通之后可以调整batch size看一下显存瓶颈的最大batch size


### Bandwidth

通过Nsight Compute查看实际DRAM数据通信总量。