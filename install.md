# 环境问题

## CUDA + cuDNN

[教程](https://blog.csdn.net/qq_40968179/article/details/128996692)

- 检查电脑版本 nvidia-msi；
- 下载对应版本CUDA，注意与pytorch版本支持对应；
  ```bash
  # CUDA 11.8
  pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118

  ```
- 下载cuDNN