import numpy as np
import json
import re


def quaternion_to_axis(q):
    """将四元数转换为旋转轴（单位方向向量）"""
    # 四元数 [x, y, z, w] 或 [w, x, y, z]
    # 从数据看，sturm使用的是[x, y, z, w]格式
    if len(q) == 4:
        x, y, z, w = q
        # 确保四元数是单位四元数
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm

        # 提取旋转轴
        sin_half_angle = np.sqrt(x * x + y * y + z * z)
        if sin_half_angle < 1e-6:  # 接近零旋转
            return np.array([1, 0, 0])  # 任意轴

        axis = np.array([x, y, z]) / sin_half_angle
        return axis
    return None


def normalize_vector(v):
    """单位化向量"""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v
    return v / norm


def angle_between_vectors(v1, v2):
    """计算两个向量之间的角度（弧度）"""
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)

    # 考虑方向性，取最小角度
    dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    dot_product_neg = np.clip(np.dot(v1_norm, -v2_norm), -1.0, 1.0)

    angle1 = np.arccos(np.abs(dot_product))
    angle2 = np.arccos(np.abs(dot_product_neg))

    return min(angle1, angle2)


def distance_between_lines(point1, dir1, point2, dir2):
    """计算两条直线之间的距离"""
    dir1_norm = normalize_vector(dir1)
    dir2_norm = normalize_vector(dir2)

    # 向量从line1上的点指向line2上的点
    w = np.array(point1) - np.array(point2)

    # 两直线方向向量的叉积
    cross_dirs = np.cross(dir1_norm, dir2_norm)
    cross_norm = np.linalg.norm(cross_dirs)

    if cross_norm < 1e-6:  # 平行直线
        # 计算点到直线的距离
        w_cross_dir1 = np.cross(w, dir1_norm)
        return np.linalg.norm(w_cross_dir1)
    else:  # 异面直线
        return abs(np.dot(w, cross_dirs)) / cross_norm


# 解析数据
data_text = """
open drawer：groundtruth："axis": [
     -0.9697839468107998,
     0.2420309164696727,
     0.030661571729421386
   ],
 my method :Axis: [-0.9699195027351379, 0.2418399155139923, 0.0277404524385929]
sturm:Axis [0.969920, -0.241839, -0.027738]

Open refridge groundtruth：  "axis": [
     0.01845365535559069,
     0.012822284654900052,
     0.9997474939304655
   ],
   "pivot": [
     0.4029322715340261,
     -1.1013132738663365,
     1.1190551332501593
   ],
 my method :Axis: [0.0184536557644606, 0.0128222843632102, 0.9997475147247314]
pivot: [0.3844105005264282, -1.1114546060562134, 0.7473650574684143]

sturm:Axis Quaternion:[-0.009084, 0.007325, -0.264831, 0.964224]

pivot [0.384808, -1.132571, 0.450645]

close refridge groundtruth：  "axis": [
     0.01845365535559069,
     0.012822284654900052,
     0.9997474939304655
   ],
   "pivot": [
     0.4029322715340261,
     -1.1013132738663365,
     1.1190551332501593
   ],
my method : wrong
sturm:Axis Quaternion: [-0.011137, 0.001500, -0.737707, 0.675028]
 pivot [0.356209, -1.106706, 0.418062]

Open gasstove groundtruth：  "axis": [
     -0.25076136242165215,
     -0.9675979725539509,
     0.029545029848670586
   ],
   "pivot": [
     -1.071033719249306,
     -0.7808217390379121,
     0.820246759610822
   ]

my method :Axis:  [-0.2507613599300385, -0.9675979614257812, 0.0295450277626514]

pivot: [-1.0710337162017822, -0.7808217406272888, 0.8202467560768127]

sturm:wrong

Close gasstove groundtruth：  "axis": [
     -0.25076136242165215,
     -0.9675979725539509,
     0.029545029848670586
   ],
   "pivot": [
     -1.071033719249306,
     -0.7808217390379121,
     0.820246759610822
   ]

 my method :Axis: [0.2507613897323608, 0.9675980210304260, -0.0295450575649738]

pivot:[-1.0712608098983765, -0.7812873125076294, 0.8195481300354004]

sturm: wrong

Open microwave groundtruth：  "axis": [
     -0.007626844168815916,
     -0.024346535164240672,
     -1.680921042346571
   ],
   "pivot": [
     0.8239041571448611,
     -1.3102364948144283,
     1.383042239874449
   ]

 my method :Axis: [0.0045367744751275, 0.0144823715090752, 0.9998848438262939]

pivot:  [0.8240529298782349, -1.3110175132751465, 1.3581254482269287]

sturm:Axis Quaternion: [-0.001159, 0.008750, 0.624192, 0.781221]

 pivot [0.128214, -1.150922, 1.263329]

Close microwave groundtruth：  "axis": [
     -0.007626844168815916,
     -0.024346535164240672,
     -1.680921042346571
   ],
   "pivot": [
     0.8239041571448611,
     -1.3102364948144283,
     1.383042239874449
   ]

 my method :Axis:  [0.0045367865823209, 0.0144823715090752, 0.9998848438262939]

pivot:  [0.8240140676498413, -1.3112387657165527, 1.3451364040374756]

sturm:Axis Quaternion: [-0.006430, 0.004046, 0.256441, 0.966530]

 pivot[0.752443, -1.186982, 1.255471]

Open washmachine groundtruth： "axis": [
 -0.0032475388711064625,
 -0.005552328075020032,
 -1.0002714472900982
],
"pivot": [
 -1.31867557373737,
 -0.7040606146473997,
 0.4345896043142765
]

 my method :Axis:  [-0.0013785789487883, 0.0048441379331052, 0.9999873042106628]

pivot:  [-1.3138686418533325, -0.7073386907577515, 0.4789380729198456]

sturm:Axis Quaternion: [-0.002496, 0.000335, 0.399230, 0.916847]

 pivot [-1.329630, -0.676689, 0.461210]

Close washmachine groundtruth： "axis": [
 -0.0032475388711064625,
 -0.005552328075020032,
 -1.0002714472900982
],
"pivot": [
 -1.31867557373737,
 -0.7040606146473997,
 0.4345896043142765
]

 my method :Axis:  [-0.0013785597402602, 0.0048441300168633, 0.9999873042106628]

pivot:  [-1.3169221878051758, -0.7077842950820923, 0.4788598418235779]

sturm:Axis Quaternion:[-0.002418, 0.000709, 0.533749, 0.845639]

 pivot [-1.221029, -0.783668, 0.471952]


Open trashbin groundtruth： "axis": [
     0.2743059868873246,
     0.9613826083564436,
     -0.02235410270022261
   ],
   "pivot": [
     1.7295686496677478,
     -0.07319971922852087,
     0.9065431070083432
   ]

 my method :Axis:  [0.2671099305152893, 0.9636483788490295, -0.0058460705913603]

pivot:  [1.7380524873733521, -0.0762164145708084, 0.8991466760635376]

sturm:wrong

sit on chair groundtruth： "axis": (0,0,1)

 my method :Axis:   [-0.0061969729140401, -0.0515742897987366, -0.9986500144004822]

sturm:wrong

sit up chair groundtruth： "axis": (0,0,1)

 my method :Axis:  [0.0046485681086779, -0.0207716114819050, 0.9997735023498535]

sturm:wrong
"""

# 手动解析数据（由于格式不规则，使用手动方式）
actions_data = {
    "open_drawer": {
        "groundtruth": {
            "axis": [-0.9697839468107998, 0.2420309164696727, 0.030661571729421386]
        },
        "my_method": {
            "axis": [-0.9699195027351379, 0.2418399155139923, 0.0277404524385929]
        },
        "sturm": {
            "axis": [0.969920, -0.241839, -0.027738]
        }
    },
    "open_refridge": {
        "groundtruth": {
            "axis": [0.01845365535559069, 0.012822284654900052, 0.9997474939304655],
            "pivot": [0.4029322715340261, -1.1013132738663365, 1.1190551332501593]
        },
        "my_method": {
            "axis": [0.0184536557644606, 0.0128222843632102, 0.9997475147247314],
            "pivot": [0.3844105005264282, -1.1114546060562134, 0.7473650574684143]
        },
        "sturm": {
            "quaternion": [-0.009084, 0.007325, -0.264831, 0.964224],
            "pivot": [0.384808, -1.132571, 0.450645]
        }
    },
    "close_refridge": {
        "groundtruth": {
            "axis": [0.01845365535559069, 0.012822284654900052, 0.9997474939304655],
            "pivot": [0.4029322715340261, -1.1013132738663365, 1.1190551332501593]
        },
        "my_method": "wrong",
        "sturm": {
            "quaternion": [-0.011137, 0.001500, -0.737707, 0.675028],
            "pivot": [0.356209, -1.106706, 0.418062]
        }
    },
    "open_gasstove": {
        "groundtruth": {
            "axis": [-0.25076136242165215, -0.9675979725539509, 0.029545029848670586],
            "pivot": [-1.071033719249306, -0.7808217390379121, 0.820246759610822]
        },
        "my_method": {
            "axis": [-0.2507613599300385, -0.9675979614257812, 0.0295450277626514],
            "pivot": [-1.0710337162017822, -0.7808217406272888, 0.8202467560768127]
        },
        "sturm": "wrong"
    },
    "close_gasstove": {
        "groundtruth": {
            "axis": [-0.25076136242165215, -0.9675979725539509, 0.029545029848670586],
            "pivot": [-1.071033719249306, -0.7808217390379121, 0.820246759610822]
        },
        "my_method": {
            "axis": [0.2507613897323608, 0.9675980210304260, -0.0295450575649738],
            "pivot": [-1.0712608098983765, -0.7812873125076294, 0.8195481300354004]
        },
        "sturm": "wrong"
    },
    "open_microwave": {
        "groundtruth": {
            "axis": [-0.007626844168815916, -0.024346535164240672, -1.680921042346571],
            "pivot": [0.8239041571448611, -1.3102364948144283, 1.383042239874449]
        },
        "my_method": {
            "axis": [0.0045367744751275, 0.0144823715090752, 0.9998848438262939],
            "pivot": [0.8240529298782349, -1.3110175132751465, 1.3581254482269287]
        },
        "sturm": {
            "quaternion": [-0.001159, 0.008750, 0.624192, 0.781221],
            "pivot": [0.128214, -1.150922, 1.263329]
        }
    },
    "close_microwave": {
        "groundtruth": {
            "axis": [-0.007626844168815916, -0.024346535164240672, -1.680921042346571],
            "pivot": [0.8239041571448611, -1.3102364948144283, 1.383042239874449]
        },
        "my_method": {
            "axis": [0.0045367865823209, 0.0144823715090752, 0.9998848438262939],
            "pivot": [0.8240140676498413, -1.3112387657165527, 1.3451364040374756]
        },
        "sturm": {
            "quaternion": [-0.006430, 0.004046, 0.256441, 0.966530],
            "pivot": [0.752443, -1.186982, 1.255471]
        }
    },
    "open_washmachine": {
        "groundtruth": {
            "axis": [-0.0032475388711064625, -0.005552328075020032, -1.0002714472900982],
            "pivot": [-1.31867557373737, -0.7040606146473997, 0.4345896043142765]
        },
        "my_method": {
            "axis": [-0.0013785789487883, 0.0048441379331052, 0.9999873042106628],
            "pivot": [-1.3138686418533325, -0.7073386907577515, 0.4789380729198456]
        },
        "sturm": {
            "quaternion": [-0.002496, 0.000335, 0.399230, 0.916847],
            "pivot": [-1.329630, -0.676689, 0.461210]
        }
    },
    "close_washmachine": {
        "groundtruth": {
            "axis": [-0.0032475388711064625, -0.005552328075020032, -1.0002714472900982],
            "pivot": [-1.31867557373737, -0.7040606146473997, 0.4345896043142765]
        },
        "my_method": {
            "axis": [-0.0013785597402602, 0.0048441300168633, 0.9999873042106628],
            "pivot": [-1.3169221878051758, -0.7077842950820923, 0.4788598418235779]
        },
        "sturm": {
            "quaternion": [-0.002418, 0.000709, 0.533749, 0.845639],
            "pivot":  [-1.221029, -0.783668, 0.471952]
        }
    },
    "open_trashbin": {
        "groundtruth": {
            "axis": [0.2743059868873246, 0.9613826083564436, -0.02235410270022261],
            "pivot": [1.7295686496677478, -0.07319971922852087, 0.9065431070083432]
        },
        "my_method": {
            "axis": [0.2671099305152893, 0.9636483788490295, -0.0058460705913603],
            "pivot": [1.7380524873733521, -0.0762164145708084, 0.8991466760635376]
        },
        "sturm": "wrong"
    },
    "sit_on_chair": {
        "groundtruth": {
            "axis": [0, 0, 1]
        },
        "my_method": {
            "axis": [-0.0061969729140401, -0.0515742897987366, -0.9986500144004822]
        },
        "sturm": "wrong"
    },
    "sit_up_chair": {
        "groundtruth": {
            "axis": [0, 0, 1]
        },
        "my_method": {
            "axis": [0.0046485681086779, -0.0207716114819050, 0.9997735023498535]
        },
        "sturm": "wrong"
    }
}


def format_to_6_decimals(data):
    """将数据格式化为6位小数"""
    if isinstance(data, (list, tuple)):
        return [round(float(x), 6) for x in data]
    elif isinstance(data, (int, float)):
        return round(float(data), 6)
    elif isinstance(data, dict):
        return {k: format_to_6_decimals(v) for k, v in data.items()}
    else:
        return data


# 计算误差
results = {}

for action_name, action_data in actions_data.items():
    action_result = {
        "groundtruth": {
            "axis": format_to_6_decimals(action_data["groundtruth"]["axis"])
        },
        "my_method": {},
        "sturm": {}
    }

    # 添加groundtruth的pivot（如果有）
    if "pivot" in action_data["groundtruth"]:
        action_result["groundtruth"]["pivot"] = format_to_6_decimals(action_data["groundtruth"]["pivot"])

    gt_axis = np.array(action_data["groundtruth"]["axis"])
    gt_axis_norm = normalize_vector(gt_axis)
    gt_pivot = action_data["groundtruth"].get("pivot")

    # 处理 my_method
    if action_data["my_method"] == "wrong":
        action_result["my_method"]["status"] = "wrong"
    else:
        my_axis = np.array(action_data["my_method"]["axis"])
        my_axis_norm = normalize_vector(my_axis)

        # 添加my_method的原始数据
        action_result["my_method"]["axis"] = format_to_6_decimals(action_data["my_method"]["axis"])
        if "pivot" in action_data["my_method"]:
            action_result["my_method"]["pivot"] = format_to_6_decimals(action_data["my_method"]["pivot"])

        # 计算角度误差（转换为度）
        angle_error = angle_between_vectors(gt_axis_norm, my_axis_norm)
        action_result["my_method"]["axis_angle_error_deg"] = round(np.degrees(angle_error), 6)

        # 如果有pivot，计算距离误差
        if gt_pivot and "pivot" in action_data["my_method"]:
            my_pivot = np.array(action_data["my_method"]["pivot"])
            line_distance = distance_between_lines(gt_pivot, gt_axis_norm, my_pivot, my_axis_norm)
            action_result["my_method"]["axis_distance_error"] = round(line_distance, 6)

    # 处理 sturm
    if action_data["sturm"] == "wrong":
        action_result["sturm"]["status"] = "wrong"
    else:
        if "quaternion" in action_data["sturm"]:
            # 添加原始四元数
            action_result["sturm"]["quaternion"] = format_to_6_decimals(action_data["sturm"]["quaternion"])
            # 将四元数转换为轴
            sturm_axis = quaternion_to_axis(action_data["sturm"]["quaternion"])
            if sturm_axis is not None:
                action_result["sturm"]["axis"] = format_to_6_decimals(sturm_axis.tolist())
        else:
            sturm_axis = np.array(action_data["sturm"]["axis"])
            action_result["sturm"]["axis"] = format_to_6_decimals(action_data["sturm"]["axis"])

        # 添加sturm的pivot（如果有）
        if "pivot" in action_data["sturm"]:
            action_result["sturm"]["pivot"] = format_to_6_decimals(action_data["sturm"]["pivot"])

        if sturm_axis is not None:
            sturm_axis_norm = normalize_vector(sturm_axis)

            # 计算角度误差（转换为度）
            angle_error = angle_between_vectors(gt_axis_norm, sturm_axis_norm)
            action_result["sturm"]["axis_angle_error_deg"] = round(np.degrees(angle_error), 6)

            # 如果有pivot，计算距离误差
            if gt_pivot and "pivot" in action_data["sturm"]:
                sturm_pivot = np.array(action_data["sturm"]["pivot"])
                line_distance = distance_between_lines(gt_pivot, gt_axis_norm, sturm_pivot, sturm_axis_norm)
                action_result["sturm"]["axis_distance_error"] = round(line_distance, 6)

    results[action_name] = action_result

# 输出JSON格式结果
print(json.dumps(results, indent=2, ensure_ascii=False))

# 保存到文件
with open('axis_error_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n结果已保存到 axis_error_analysis.json 文件中")