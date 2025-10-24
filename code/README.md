## GPU 环境

运行环境统一为：

hendrixfut01fl


## 测试

latency统一测试参数：

num_attention_heads=16

hidden_size=512

hidden_dim=512/16=32

seqlen=1024

batch_size=1


跑通之后可以调整batch size看一下显存瓶颈的最大batch size
