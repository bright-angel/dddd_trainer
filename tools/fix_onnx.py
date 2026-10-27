"""把 dddd_trainer 导出的 ONNX 修复成 ddddocr 可直接使用的单文件模型。

修复三件事（旧导出方式的产物在 ddddocr 里会退化成一个乱码字符）：
1. 权重内联：torch 新导出器会把权重写到同名 <model>.onnx.data，单独拷走 .onnx 会加载失败；
2. 输出改回 logits：Net.forward 末尾的 ArgMax 输出的是类别索引，而 ddddocr 需要
   [time_step, batch, charset] 的 logits，由它自己 argmax + CTC 解码；
3. 折算输入归一化：训练用了 mean/std 归一化，ddddocr 只把像素除以 255，
   所以在图开头补一层 (x - mean) / std。

用法:
    uv run tools/fix_onnx.py projects/<name>/models/xxx.onnx [--out xxx_fixed.onnx]
"""

import argparse
import os
import sys

import onnx
import onnx.helper as helper
from onnx import TensorProto

NORMALIZE = {1: ([0.456], [0.224]), 3: ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])}


def cut_argmax(model):
    """删掉图尾部的 ArgMax(+随后的 Transpose)，改为输出它的输入张量（logits）。"""
    graph = model.graph
    producers = {out: node for node in graph.node for out in node.output}
    argmax = next((node for node in graph.node if node.op_type == "ArgMax"), None)
    if argmax is None:
        print("图中没有 ArgMax，输出已是 logits，跳过改输出")
        return model
    logits = argmax.input[0]
    axis = next(a.i for a in argmax.attribute if a.name == "axis")

    used = set()
    stack = [logits]
    while stack:
        name = stack.pop()
        if name in used:
            continue
        used.add(name)
        node = producers.get(name)
        if node is not None:
            stack.extend(node.input)

    kept = [node for node in graph.node if any(out in used for out in node.output)]
    del graph.node[:]
    graph.node.extend(kept)
    for node in graph.node:
        for idx, out in enumerate(node.output):
            if out == logits:
                node.output[idx] = "output"

    classes = next((init.dims[0] for init in graph.initializer
                    if init.name.endswith("fc.bias")), None)
    if classes is None:
        raise SystemExit("找不到 fc.bias，无法推断字符集长度")
    shape = ["seqlen", 1, classes] if axis == 2 else ["batch", classes]
    del graph.output[:]
    graph.output.extend([helper.make_tensor_value_info("output", TensorProto.FLOAT, shape)])
    del graph.value_info[:]
    return model


def fold_normalize(model):
    """在图开头补 (x - mean) / std，抵消训练做过而 ddddocr 没做的归一化。"""
    graph = model.graph
    channel = int(graph.input[0].type.tensor_type.shape.dim[1].dim_value)
    if channel not in NORMALIZE:
        raise SystemExit("输入通道 {} 未知，无法折算归一化".format(channel))
    mean, std = NORMALIZE[channel]
    src = graph.input[0].name
    graph.initializer.extend([
        helper.make_tensor("dddd_mean", TensorProto.FLOAT, [1, len(mean), 1, 1], mean),
        helper.make_tensor("dddd_std", TensorProto.FLOAT, [1, len(std), 1, 1], std),
    ])
    graph.node.insert(0, helper.make_node(
        "Sub", [src, "dddd_mean"], ["dddd_normalized"], name="dddd_normalize"))
    graph.node.insert(1, helper.make_node(
        "Div", ["dddd_normalized", "dddd_std"], ["dddd_standardized"], name="dddd_standardize"))
    for node in graph.node[2:]:
        for idx, inp in enumerate(node.input):
            if inp == src:
                node.input[idx] = "dddd_standardized"
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--out")
    args = parser.parse_args()

    src = os.path.abspath(args.model)
    dst = os.path.abspath(args.out or os.path.splitext(src)[0] + "_fixed.onnx")

    model = onnx.load(src, load_external_data=True)
    model = cut_argmax(model)
    model = fold_normalize(model)
    model.ir_version = 7

    current = model.opset_import[0].version
    if current > 12:
        try:
            model = onnx.version_converter.convert_version(model, 12)
            print("opset {} -> 12 降级成功".format(current))
        except Exception as error:  # noqa: BLE001
            print("opset 降级到 12 失败，保留 {}: {}".format(current, str(error)[:100]))

    onnx.checker.check_model(model)
    onnx.save(model, dst, save_as_external_data=False)
    print("已写出单文件模型: {} ({:.1f} MB, opset {}, ir {})".format(
        dst, os.path.getsize(dst) / 1024 / 1024, model.opset_import[0].version, model.ir_version))
    return 0


if __name__ == "__main__":
    sys.exit(main())
