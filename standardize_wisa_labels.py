#!/usr/bin/env python3
"""
将 WISA 数据集的 CSV 标签映射到标准 10 分类，并整合 q0/q1/q2 的多标签信息。
"""

import csv
import sys
from collections import Counter

# 标准 10 分类
PHENOMENON_LABELS = [
    "Rigid Body", "Elastic", "Fluid", "Compressible Flow", "Phase Change",
    "Collision/Contact", "Granular", "Fracture", "Thermal", "Optical",
]

# WISA 原始标签到标准标签的映射
WISA_TO_STANDARD = {
    'rigid body motion': 'Rigid Body',
    'elastic motion': 'Elastic',
    'liquid motion': 'Fluid',
    'gas motion': 'Compressible Flow',
    'collision': 'Collision/Contact',
    'deformation': 'Elastic',
    'melting': 'Phase Change',
    'solidification': 'Phase Change',
    'vaporization': 'Phase Change',
    'liquefaction': 'Phase Change',
    'combustion': 'Thermal',
    'explosion': 'Phase Change',
    'reflection': 'Optical',
    'refraction': 'Optical',
    'scattering': 'Optical',
    'interference and diffraction': 'Optical',
    'unnatural light source': 'Optical',
}

# q0 动态现象关键词映射
Q0_KEYWORDS = {
    'rigid body motion': 'Rigid Body',
    'elastic motion': 'Elastic',
    'liquid motion': 'Fluid',
    'gas motion': 'Compressible Flow',
    'collision': 'Collision/Contact',
    'deformation': 'Elastic',
}

# q1 热力学现象关键词映射
Q1_KEYWORDS = {
    'melting': 'Phase Change',
    'solidification': 'Phase Change',
    'vaporization': 'Phase Change',
    'liquefaction': 'Phase Change',
    'combustion': 'Thermal',
    'explosion': 'Phase Change',
}

# q2 光学现象关键词映射
Q2_KEYWORDS = {
    'reflection': 'Optical',
    'refraction': 'Optical',
    'scattering': 'Optical',
    'interference and diffraction': 'Optical',
    'unnatural light source': 'Optical',
}


def extract_labels_from_field(field_value, keyword_map):
    """从字段值中提取匹配的关键词对应的标签"""
    labels = set()
    field_lower = field_value.lower()

    for keyword, standard_label in keyword_map.items():
        if keyword in field_lower:
            labels.add(standard_label)

    return labels


def process_row(row):
    """处理单行数据，返回新的标准化标签"""
    # 获取原始标签
    original_label = row.get('label', '').strip().lower()

    # 获取 q0, q1, q2
    q0 = row.get('q0', '').strip()
    q1 = row.get('q1', '').strip()
    q2 = row.get('q2', '').strip()

    # 收集所有标签
    all_labels = set()

    # 1. 从原始 label 映射
    if original_label in WISA_TO_STANDARD:
        all_labels.add(WISA_TO_STANDARD[original_label])

    # 2. 从 q0 提取（排除 "no obvious" 的情况）
    if 'no obvious' not in q0.lower():
        labels_from_q0 = extract_labels_from_field(q0, Q0_KEYWORDS)
        all_labels.update(labels_from_q0)

    # 3. 从 q1 提取（排除 "no obvious" 的情况）
    if 'no obvious' not in q1.lower():
        labels_from_q1 = extract_labels_from_field(q1, Q1_KEYWORDS)
        all_labels.update(labels_from_q1)

    # 4. 从 q2 提取（排除 "no obvious" 的情况）
    if 'no obvious' not in q2.lower():
        labels_from_q2 = extract_labels_from_field(q2, Q2_KEYWORDS)
        all_labels.update(labels_from_q2)

    # 如果没有提取到任何标签，使用默认值
    if not all_labels:
        all_labels = {'Fluid'}

    # 按照标准顺序排序，确保输出一致性
    ordered_labels = []
    for std_label in PHENOMENON_LABELS:
        if std_label in all_labels:
            ordered_labels.append(std_label)

    return ', '.join(ordered_labels)


def main():
    input_file = '/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_new.csv'
    output_file = '/home/dataset-assist-0/algorithm/cong.wang/cvpr/wisa-dataset/WISA-80K/data/metadata_standardized.csv'

    print(f"Processing {input_file}...")

    # 统计信息
    original_label_counts = Counter()
    new_label_counts = Counter()

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(reader):
            # 统计原始标签
            original_label = row.get('label', '').strip()
            original_label_counts[original_label] += 1

            # 生成新的标准化标签
            new_label = process_row(row)
            new_label_counts[new_label] += 1

            # 更新行数据
            row['label'] = new_label
            writer.writerow(row)

            if (i + 1) % 10000 == 0:
                print(f"  Processed {i + 1} rows...")

    print(f"\nDone! Output saved to {output_file}")

    print("\n" + "="*60)
    print("Original label distribution:")
    for label, count in original_label_counts.most_common():
        print(f"  {count:6d}: {label}")

    print("\n" + "="*60)
    print("New standardized label distribution:")
    for label, count in new_label_counts.most_common():
        print(f"  {count:6d}: {label}")

    print("\n" + "="*60)
    print(f"Total rows processed: {sum(original_label_counts.values())}")
    print(f"Unique label combinations: {len(new_label_counts)}")


if __name__ == '__main__':
    main()
