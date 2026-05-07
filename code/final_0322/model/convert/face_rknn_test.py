import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

# ================= 1. 路径配置 =================
RKNN_MODEL_PATH = './facenet.rknn'
FACE1_PATH = '/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/face_image/face1.png'
FACE2_PATH = '/home/test/openbot_test_zhenghang/model_0320/openbot_validify_code/final_0322/model/convert/face_image/face2.png'

SIM_THRESHOLD = 0.7


def check_file(path, desc):
    if not os.path.exists(path):
        raise FileNotFoundError(f'❌ 找不到{desc}: {path}')


def preprocess_image(image_path):
    """
    输入改成 RKNN 当前要求的 4D NCHW:
    [1, 3, 160, 160]

    注意：
    不要再手工做 (x - 127.5) / 128.0
    因为这些已经在 RKNN 转换时通过 mean/std 配进模型了。
    """
    check_file(image_path, '图片文件')

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'❌ 图片读取失败，请检查文件是否损坏: {image_path}')

    resized = cv2.resize(img, (160, 160))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = np.transpose(rgb, (2, 0, 1))      # HWC -> CHW
    tensor = np.expand_dims(tensor, axis=0)    # CHW -> NCHW
    tensor = np.ascontiguousarray(tensor, dtype=np.uint8)

    return tensor


def get_attr_value(attr, names, default=None):
    if attr is None:
        return default
    for name in names:
        if isinstance(attr, dict) and name in attr:
            return attr[name]
        if hasattr(attr, name):
            return getattr(attr, name)
    return default


def try_query_output_attr(rknn, output_index=0):
    if hasattr(rknn, 'query'):
        possible_const_names = ['QUERY_OUTPUT_ATTR', 'RKNN_QUERY_OUTPUT_ATTR']
        for const_name in possible_const_names:
            query_const = getattr(type(rknn), const_name, None)
            if query_const is None:
                query_const = getattr(rknn, const_name, None)
            if query_const is not None:
                try:
                    return rknn.query(query_const, output_index)
                except Exception:
                    pass
    return None


def dequantize_if_needed(output_array, output_attr):
    arr = np.asarray(output_array)

    if arr.dtype not in (np.int8, np.uint8, np.int16):
        return arr.astype(np.float32), False, None, None

    scale = get_attr_value(output_attr, ['scale', 'qnt_scale'])
    zp = get_attr_value(output_attr, ['zp', 'zero_point', 'qnt_zp'], 0)

    if scale is None:
        return arr.astype(np.float32), False, None, None

    arr = arr.astype(np.float32)
    arr = (arr - float(zp)) * float(scale)
    return arr, True, float(scale), int(zp)


def postprocess_embedding(output_array, output_attr=None):
    """
    补回之前删掉的部分：
    1. 反量化（如果能取到量化参数）
    2. 1x512x1x1 -> 512
    3. L2 normalize
    """
    output_fp32, dequant_ok, scale, zp = dequantize_if_needed(output_array, output_attr)

    vector = output_fp32.reshape(-1).astype(np.float32)

    norm = np.linalg.norm(vector)
    if norm < 1e-9:
        normed = vector
    else:
        normed = vector / norm

    debug_info = {
        'raw_dtype': str(np.asarray(output_array).dtype),
        'raw_shape': tuple(np.asarray(output_array).shape),
        'dequantized': dequant_ok,
        'scale': scale,
        'zp': zp,
        'feature_dim': int(normed.shape[0]),
        'l2_norm_before': float(norm),
        'l2_norm_after': float(np.linalg.norm(normed)),
    }
    return normed, debug_info


def cosine_similarity(vec1, vec2):
    vec1 = vec1.astype(np.float32).reshape(-1)
    vec2 = vec2.astype(np.float32).reshape(-1)

    dot_product = np.dot(vec1, vec2)
    norm_v1 = np.linalg.norm(vec1)
    norm_v2 = np.linalg.norm(vec2)

    if norm_v1 < 1e-9 or norm_v2 < 1e-9:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))


def load_rknn_model(rknn_path):
    check_file(rknn_path, 'RKNN 模型文件')

    print('🔄 正在加载 RKNN 模型...')
    rknn = RKNNLite()

    ret = rknn.load_rknn(rknn_path)
    if ret != 0:
        raise RuntimeError(f'❌ load_rknn 失败，返回值: {ret}')

    print('🔄 正在初始化 RKNN 运行时...')
    try:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO)
    except Exception:
        ret = rknn.init_runtime()

    if ret != 0:
        raise RuntimeError(f'❌ init_runtime 失败，返回值: {ret}')

    print('✅ RKNN 模型加载完毕！')
    return rknn


def extract_feature_rknn(image_path, rknn, output_index=0):
    input_tensor = preprocess_image(image_path)
    print(f'   输入 shape: {input_tensor.shape}, dtype: {input_tensor.dtype}')

    outputs = rknn.inference(inputs=[input_tensor], data_format=['nchw'])
    if outputs is None or len(outputs) == 0:
        raise RuntimeError('❌ RKNN 推理没有返回输出')

    raw_output = outputs[output_index]
    output_attr = try_query_output_attr(rknn, output_index=output_index)
    feature, debug_info = postprocess_embedding(raw_output, output_attr)

    return feature, debug_info


if __name__ == '__main__':
    print('\n' + '=' * 60)
    print('🎯 FaceNet RKNN 模型精度验证（含后处理恢复）')
    print('=' * 60)

    rknn = None
    try:
        rknn = load_rknn_model(RKNN_MODEL_PATH)

        print(f'\n📷 提取特征 1: {os.path.basename(FACE1_PATH)}')
        vec1, info1 = extract_feature_rknn(FACE1_PATH, rknn)
        print(f'   输出信息: dtype={info1["raw_dtype"]}, shape={info1["raw_shape"]}, '
              f'dequantized={info1["dequantized"]}, dim={info1["feature_dim"]}')
        if info1["dequantized"]:
            print(f'   量化参数: scale={info1["scale"]}, zp={info1["zp"]}')

        print(f'\n📷 提取特征 2: {os.path.basename(FACE2_PATH)}')
        vec2, info2 = extract_feature_rknn(FACE2_PATH, rknn)
        print(f'   输出信息: dtype={info2["raw_dtype"]}, shape={info2["raw_shape"]}, '
              f'dequantized={info2["dequantized"]}, dim={info2["feature_dim"]}')
        if info2["dequantized"]:
            print(f'   量化参数: scale={info2["scale"]}, zp={info2["zp"]}')

        sim = cosine_similarity(vec1, vec2)

        print('\n' + '=' * 60)
        print(f'🧠 余弦相似度得分: {sim:.4f}')
        print('=' * 60)

        if sim >= SIM_THRESHOLD:
            print(f'✅ 结论: 相似度 >= {SIM_THRESHOLD:.1f}，当前判定为【同一个人】。')
        else:
            print(f'❌ 结论: 相似度 < {SIM_THRESHOLD:.1f}，当前判定为【不同的人】。')
            print('💡 提示: 量化模型的阈值可能需要重新标定。')

    except Exception as e:
        print(f'\n💥 运行发生异常: {e}')

    finally:
        if rknn is not None:
            try:
                rknn.release()
            except Exception:
                pass
