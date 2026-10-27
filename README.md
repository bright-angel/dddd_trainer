# dddd_trainer 带带弟弟OCR训练工具 （2025.01.18项目即将更新，将支持完善的可视化训练和新的训练策略）

### 带带弟弟OCR所用的训练工具今天正式开源啦！ [ddddocr](https://github.com/sml2h3/ddddocr)

### 项目仅支持N卡训练，A卡或其他卡就先别看啦

### 项目基于Pytorch进行开发，支持cnn与crnn进行训练、断点恢复、自动导出onnx模型，并同时支持无缝使用[ddddocr](https://github.com/sml2h3/ddddocr) 与 [ocr_api_server](https://gitee.com/fkgeek/ocr_api_server) 的无缝部署

### 训练环境支持

Windows/Linux

Macos仅支持cpu训练

## 1、深度学习必备环境配置（非仅本项目要求，而是所有深度学习项目要求，cpu训练除外）

### 开始本教程前请先前往[pytorch](https://pytorch.org/get-started/locally/) 官网查看自己系统与硬件支持的pytorch版本，注意30系列之前的N卡，如2080Ti等请选择cuda11以下的版本（例：CUDA 10.2），如果为30系N卡，仅支持CUDA 11版本，请选择CUDA 11以上版本（例：CUDA 11.3），然后根据选择的条件显示的pytorch安装命令完成pytorch安装，由于pytorch的版本更新速度导致很多pypi源仅缓存了cpu版本，CUDA版本需要自己在官网安装。

### 安装CUDA和CUDNN

根据自己显卡型号与系统选择

[cuda](https://developer.nvidia.com/cuda-downloads)

[cudnn](https://developer.nvidia.com/zh-cn/cudnn)

注意cudnn支持的cuda版本号要与你安装的cuda版本号对应，不同版本的cuda支持的显卡不一样，<b>20系无脑选择10.2版本cuda，30系无脑选择11.3版本cuda</b>,这里有啥问题就百度吧，算是一个基础问题。

### ONNX 导出相关的版本要求（务必看完再训练）

| 依赖 | 要求 | 说明 |
| --- | --- | --- |
| Python | 3.8 ~ 3.12 | 3.13 上部分 wheel 仍不齐全 |
| torch | 建议 2.5 以上 | torch>=2.6 起 `torch.onnx.export` 默认走 dynamo 导出器，会把权重写到同名 `<模型>.onnx.data` 外置文件，单独拷走 `.onnx` 无法加载。本项目导出已固定 `dynamo=False`，始终输出单文件模型 |
| onnx | >= 1.14 | 导出兜底内联权重与 `tools/fix_onnx.py` 修复脚本需要 |
| onnxruntime | >= 1.14 | 部署侧（ddddocr / ocr_api_server）需要，导出的模型为 opset 12 + ir_version 7，兼容该版本及以上 |

导出与部署两侧还有一条硬性约定：<b>训练时的输入归一化必须和推理时一致</b>。本项目训练用 `ToTensor` + `Normalize(mean=[0.456], std=[0.224])`（3 通道为 ImageNet 的 mean/std），而 ddddocr 加载自定义模型时只做 `像素/255`，所以导出时会把 `(x - mean) / std` 折算进 ONNX 图的开头，训练侧不需要再改。

## 2、训练部分 

- 以下所有变量均以 {param} 格式代替，表示可根据自己需要修改，而使用时并不需要带上{}，如步骤创建新的训练项目，使用时可以直接写

`python app.py create test_project`

- ### 1、Clone本项目到本地

`git clone https://github.com/sml2h3/dddd_trainer.git`

- ### 2、进入项目目录并安装本项目所需依赖

`pip install -r requirements.txt -i https://pypi.douban.com/simple`

- ### 3、创建新的训练项目

`python app.py create {project_name}`

如果想要创建一个CNN的项目，则可以加上--single参数，CNN项目识别比如图片类是什么分类的情况，比如图片上只有一个字，识别这张图是什么字（图上有多个字的不要用CNN模式），又比如分辨图片里是狮子还是兔子用CNN模式比较合适，大多数OCR需求请不要使用--single

`python app.py create {project_name} --single`

project_name 为项目名称，尽量不要以特殊符号命名

- ### 4、准备数据

    项目支持两种形式的数据
    
    ### A、从文件名导入
        
    图片均在同一个文件夹中，且命名为类似，其中/root/images_set为图片所在目录，可以为任意目录地址

    ```
  /root/images_set/
    |---- abcde_随机hash值.jpg
    |---- sdae_随机hash值.jpg
    |---- 酱闷肘子_随机hash值.jpg
  
  ```
    
    如下图所示

    ![image](https://cdn.wenanzhe.com/img/mkGu_000001d00f140741741ed9916240d8d5.jpg)

    那么图片命名可以是 

    `mkGu_000001d00f140741741ed9916240d8d5.jpg`

    ### 为考虑各种情况，dddd_trainer不会自动去处理大小写问题，如果想训练大小写，则在样本标注时就需要自己标注好大小写，如上面例子

    ### B、从文件中导入

    受限于可能样本组织形式或者特殊字符，本项目支持从txt文档中导入数据，数据集目录必须包含有`labels.txt`文件和`images`文件夹, 其中/root/images_set为图片所在目录，可以为任意目录地址
    
    `labels.txt`文件中包含了所有在`/root/images_set/images`目录下基于`/root/images_set/images`的图片相对路径，`/root/images_set/images`下可以有目录。

    #### 当然，在这种模式下，图片的文件名随意，可以有具体label也可以没有，因为咱们不从这里获取图片的label

    如下所示
- 
   a.images下无目录的形式

    ```
  /root/images_set/
    |---- labels.txt
    |---- images
          |---- 随机hash值.jpg
          |---- 随机hash值.jpg
          |---- 酱闷肘子_随机hash值.jpg
  
  labels.txt文件内容为（其中\t制表符为每行文件名与label的分隔符）
  随机hash值.jpg\tabcd
  随机hash值.jpg\tsdae
  酱闷肘子_随机hash值.jpg\t酱闷肘子
  ```
  b.images下有目录的形式
    ```
  /root/images_set/
    |---- labels.txt
    |---- images
          |---- aaaa
                |---- 随机hash值.jpg
          |---- 酱闷肘子_随机hash值.jpg
  
  labels.txt文件内容为（其中\t制表符为每行文件名与label的分隔符）
  aaaa/随机hash值.jpg\tabcd
  aaaa/随机hash值.jpg\tsdae
  酱闷肘子_随机hash值.jpg\t酱闷肘子
  
  ```
  
  ### 为了新手更好的理解本部分的内容，本项目也提供了两套基础数据集提供测试

    [数据集一](https://wwm.lanzoum.com/iUyYb0b5z3lg)
    [数据集二](https://wwm.lanzoum.com/itczd0b5z3yj)
- ### 5、修改配置文件
```yaml
Model:
    CharSet: []     # 字符集，不要动，会自动生成
    ImageChannel: 1 # 图片通道数，如果你想以灰度图进行训练，则设置为1，彩图，则设置为3。如果设置为1，数据集是彩图，项目会在训练的过程中自动在内存中将读取到的彩图转为灰度图，并不需要提前自己修改并且该设置不会修改本地图片
    ImageHeight: 64 # 图片自动缩放后的高度，单位为px,高度必须为16的倍数，会自动缩放图像
    ImageWidth: -1  # 图片自动缩放后的宽度，单位为px，本项若设置为-1，将自动根据情况调整
    Word: false     # 是否为CNN模型，这里在创建项目的时候通过参数控制，不要自己修改
System:
    Allow_Ext: [jpg, jpeg, png, bmp]  # 支持的图片后缀，不满足的图片将会被自动忽略
    GPU: true                         # 是否启用GPU去训练，使用GPU训练需要参考步骤一安装好环境
    GPU_ID: 0                         # GPU设备号，0为第一张显卡
    Path: ''                          # 数据集根目录，在缓存图片步骤会自动生成，不需要自己改，除非数据集地址改了
    Project: test                     # 项目名称 也就是{project_name}
    Val: 0.03                         # 验证集的数据量比例，0.03就是3%，在缓存数据时，会自动选则3%的图片用作训练过程中的数据验证，修改本值之后需要重新缓存数据
Train:
    BATCH_SIZE: 32                                    # 训练时每一个batch_size的大小，主要取决于你的显存或内存大小，可以根据自己的情况，多测试，一般为16的倍数,如16，32，64，128
    CNN: {NAME: ddddocr}                              # 特征提取的模型，目前支持的值为ddddocr,effnetv2_l,effnetv2_m,effnetv2_xl,effnetv2_s,mobilenetv2,mobilenetv3_s,mobilenetv3_l
    DROPOUT: 0.3                                      # 非专业人员不要动
    LR: 0.01                                          # 初始学习率
    OPTIMIZER: SGD                                    # 优化器，不要动
    SAVE_CHECKPOINTS_STEP: 2000                       # 每多少step保存一次模型
    TARGET: {Accuracy: 0.97, Cost: 0.05, Epoch: 20}   # 训练结束的目标，同时满足时自动结束训练并保存onnx模型，Accuracy为需要满足的最小准确率，Cost为需要满足的最小损失，Epoch为需要满足的最小训练轮数
    TEST_BATCH_SIZE: 32                               # 测试时每一个batch_size的大小，主要取决于你的显存或内存大小，可以根据自己的情况，多测试，一般为16的倍数,如16，32，64，128
    TEST_STEP: 1000                                   # 每多少step进行一次测试


```
配置文件位于本项目根目录下`projects/{project_name}/config.yaml`

- ### 6、缓存数据

`python app.py cache {project_name} /root/images_set/`

如果是从labels.txt里面读取数据

`python app.py cache {project_name} /root/images_set/ file`

- ### 7、开始训练或恢复训练

`python app.py train {project_name}`

- ### 8、部署

训练达标后会自动把模型导出到 `projects/{project_name}/models/`，同时写出 `charsets.json`（字符集、缩放尺寸、通道数）。<b>交付时只需要 `.onnx` + `charsets.json` 两个文件</b>，`.onnx` 必须是单文件（不能带 `.onnx.data`）。

使用 ddddocr 加载：

```python
import ddddocr

ocr = ddddocr.DdddOcr(det=False, ocr=False,
                      import_onnx_path="{project_name}_{acc}_{epoch}_{step}_{time}.onnx",
                      charsets_path="charsets.json")
with open("test.png", "rb") as f:
    print(ocr.classification(f.read()))
```

ddddocr 对导入模型的约定（本项目的导出已按此产出）：

1. 输入名 `input1`，形状 `[1, channel, 64, width]`，像素值除以 255；
2. 输出名 `output`，形状 `[time_step, 1, charset_len]` 的 <b>logits</b>，argmax 与 CTC 去重由 ddddocr 自己完成；
3. `charsets.json` 里字符集的第 0 个字符必须留给 CTC blank（`Word: true` 的 CNN 模型除外）。

- ### 9、旧模型（修复前导出的）怎么救

修复前导出的模型在 ddddocr 里只会返回一个乱码字符，不用重训，跑一次修复脚本即可：

```bash
pip install onnx           # 修复脚本依赖
python tools/fix_onnx.py projects/{project_name}/models/{model}.onnx
```

脚本会输出 `{model}_fixed.onnx`，依次完成：内联外置权重、剪掉图尾部 `ArgMax` 改回输出 logits、在图开头补 `(x - mean) / std`，并把 `ir_version` 压到 7。

注意：如果历史模型是 dynamo 导出器产出的（opset 18），`onnx` 的版本转换器降不到 12，脚本会保留 opset 18 并打印提示，这种模型需要部署侧的 onnxruntime >= 1.16。

- ### 10、备注
  
  1. ddddocr crnn网络结构来自 https://github.com/meijieru/crnn.pytorch.git

## 3、ONNX 导出问题修复说明

本次修复的是"训练能达标、导出的 onnx 在 ddddocr 里识别不出来"，共三个独立原因：

| # | 现象 | 原因 | 修改 |
| --- | --- | --- | --- |
| 1 | 单独拷走 `.onnx` 报 `ModelLoadError` / `EP Error` | torch 新版导出器把权重写到同名 `.onnx.data` | `Net.export_onnx` 固定 `dynamo=False`，并保留 `Net.inline_external_data` 兜底内联 |
| 2 | 识别结果塌成一个乱码字符 | `Net.forward` 末尾做了 `ArgMax`，导出图的输出是类别索引；ddddocr 需要 logits 再自己 argmax，对 2 维索引做 `argmax(axis=-1)` 得到的是"位置"而不是类别号 | 新增 `OnnxExportNet`，导出时走 `get_features` 输出 `[time_step, batch, charset]` logits；`utils/train.py` 的动态轴相应改为 `output: {0: 'seq'}` |
| 3 | 即使输出维度正确也全错 | 训练做了 mean/std 归一化，ddddocr 只做 `像素/255`，输入分布不一致 | `OnnxExportNet` 在图开头折算 `(x - mean) / std` |

验证（16 字符集的数学公式验证码，取验证集前 150 张，走 ddddocr 完整流程）：

- 修复前：识别结果恒为一个乱码字符，正确率 0/150；
- 用 `tools/fix_onnx.py` 修复历史导出模型：150/150；
- 用修复后的导出代码重新训练导出：150/150，且为单文件、opset 12、ir_version 7。
