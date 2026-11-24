# LVC1 实验码流

字节布局：`LVC1`（4字节）+ little-endian uint32 JSON字节长度 + UTF-8 JSON + 二进制payload。
JSON包含version=1、BCTHW shape、原始numpy dtype、layout、byte_order、video元数据、quantization配置、zero_point=0、scale、storage、compression、payload_bytes、raw_payload_bytes、payload SHA256。无损另存张量原始字节SHA256。视频元数据保存源manifest、VAE标识与归一化约定。模型权重不在码流内，接收端预部署。

量化使用 qmax=2^(b-1)-1，qmin=-qmax；scale=absmax/qmax 或指定绝对值percentile/qmax，最小1e-12。np.rint ties-to-even；dequant=q*scale。Per-channel沿B,T,H,W求统计。2bit实际级别为 -scale,0,+scale，保留第四码字；映射u=q+2^(b-1)。这不是四级非对称量化。

位打包：按C连续的BCTHW顺序展开，每个符号低bit先写，流内little bit order；不足字节高位补0。符号数由shape乘积确定。INT4两个符号/字节，INT2四个符号/字节；3/5/6bit也采用连续bit打包。

无损直接保存原始dtype字节，不转换FP16。Byte shuffle按每元素字节位置分组，解码逆转。Zstd调用系统libzstd level3；zlib默认级别。所有信息均计入实际文件长度。

E1：先量化，d0=q0、dt=qt-q(t-1)，用little-endian int16保存差分并Zstd。解码int32累加恢复整数，后反量化。E2：首片与周期刷新片直接编码，其余残差使用已重建上一片预测；固定预先从原张量计算的scale保存在码流，解码无需原张量。Refresh=0表示All。

混合精度：同一LVC1外层JSON含mixed=true、每通道位宽表与子码流长度。按通道拼接独立完整LVC1子码流；子流均含scale、元数据、checksum，全部开销计入总文件长度。该格式冗余较大，结果不得假定成熟容器开销。

独立解码：

```sh
python scripts/codec_cli.py decode input.lvc restored.npy
```

编码输入为有限数值五维numpy数组；当前正式实验FP32，float16/32/64无损已测试。原生BF16张量适配NOT RUN。不支持编码时调用VAE或像素数据。单个解码文件只依赖numpy、Python标准库与系统libzstd；无需原始latent、视频或torch。输出dtype和shape同原始输入。

复现服务器：Wan2.1/.venv/bin/python -m unittest discover -s tests -v；随后依次运行run_smoke.py、supplement.py、present.py、validate.py。脚本使用现有服务器路径，不下载权重或重新生成视频。所有码率按stat完整文件计算；header_bytes包含量化side information，不能再次重复加算。混合流payload_bytes包含各子码流头部。
