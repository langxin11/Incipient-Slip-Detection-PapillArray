#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成合成传感器数据，用于在下载真实数据集（Google Drive）之前冒烟验证整套流水线。

输出目录结构（与真实数据约定一致）：
    datasets/fullset/case_1/z=0.9-v=1.0-XY.csv          <- slip 文件（无 -stop 后缀）
    datasets/fullset/case_1/z=0.9-v=1.0-XY-stop.csv     <- stop 文件（文件名以 -stop.csv 结尾）
    ...

文件名约定（流水线硬依赖，勿改格式）：
    z={浮点}-v={速度}-{方向}.csv[/-stop.csv]
    v ∈ {0.5, 1.0, 2.0} 的 stop 文件按 2100 帧停止处理，其余按 2050 帧处理。
CSV 无表头，73 列，顺序与 constants.headers 一致：time + 9 柱 × (contact, slip, DX, DY, DZ, FX, FY, FZ)。
"""
import argparse
import os

import numpy as np

try:
    import constants as CONSTANTS  # 旧平铺布局
except ImportError:
    from papillarray import constants as CONSTANTS  # 归组后的包布局


def make_sequence(rng, rows, slip_windows):
    """生成一行 CSV：平滑力信号 + 按窗口置位的 slip 标签。

    slip_windows: {pillar_idx: (start, end)}，start 前与 end 后为 0。
    """
    t = np.arange(rows) * CONSTANTS.t_interval
    cols = [t]
    for p in range(9):
        contact = np.ones(rows)
        slip = np.zeros(rows, dtype=int)
        if p in slip_windows:
            s, e = slip_windows[p]
            slip[s:e] = 1
        # 平滑低频力信号 + 微噪声，保证中值滤波/差分环节有意义
        phase = rng.uniform(0, 2 * np.pi)
        fx = 0.4 * np.sin(2 * np.pi * 1.5 * t + phase) + 0.05 * rng.standard_normal(rows)
        fy = 0.3 * np.cos(2 * np.pi * 1.2 * t + phase) + 0.05 * rng.standard_normal(rows)
        slip_force = 0.8 * (slip > 0)
        fx = fx + slip_force * rng.uniform(0.05, 0.15)
        fy = fy + slip_force * rng.uniform(-0.1, 0.1)
        dx = 0.001 * np.cumsum(fx)
        dy = 0.001 * np.cumsum(fy)
        dz = 0.0005 * np.arange(rows)
        fz = 0.5 + 0.01 * rng.standard_normal(rows)
        cols += [contact, slip, dx, dy, dz, fx, fy, fz]
    data = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    # contact/slip 写成 0/1（每柱 8 列，time 之后的第 p 柱从 1+p*8 开始）
    for p in range(9):
        data[:, 1 + p * 8] = (data[:, 1 + p * 8] > 0).astype(float)
        data[:, 2 + p * 8] = data[:, 2 + p * 8].astype(int).astype(float)
    return data


def write_csv(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, data, delimiter=',', fmt='%.6f')
    print(f"wrote {path}  shape={data.shape}")


def main():
    parser = argparse.ArgumentParser(description="生成合成 PapillArray 传感器 CSV 用于冒烟测试")
    parser.add_argument('--out', default='datasets', help="输出根目录（默认 datasets）")
    parser.add_argument('--rows', type=int, default=4200, help="每个 CSV 的行数（需 >= 4000）")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    def fullset_dir(case):
        return os.path.join(args.out, 'fullset', case)

    # case_1：一个 slip 文件 + 两种停止帧的 stop 文件（保证 2100/2050 两组都非空）
    write_csv(os.path.join(fullset_dir('case_1'), 'z=0.9-v=1.0-XY.csv'),
              make_sequence(rng, args.rows, {p: (1000 + 50 * p, 4000) for p in range(9)}))
    write_csv(os.path.join(fullset_dir('case_1'), 'z=0.9-v=1.0-XY-stop.csv'),
              make_sequence(rng, args.rows, {p: (1200 + 30 * p, 2100) for p in range(9)}))
    write_csv(os.path.join(fullset_dir('case_1'), 'z=0.9-v=0.25-XZ-stop.csv'),
              make_sequence(rng, args.rows, {}))  # 完全无滑移的 stop 文件

    # case_2：再一组不同参数
    write_csv(os.path.join(fullset_dir('case_2'), 'z=0.8-v=2.0-YZ.csv'),
              make_sequence(rng, args.rows, {p: (900 + 70 * p, 3900) for p in range(9)}))
    write_csv(os.path.join(fullset_dir('case_2'), 'z=0.8-v=2.0-YZ-stop.csv'),
              make_sequence(rng, args.rows, {p: (1100 + 40 * p, 2100) for p in range(9)}))
    write_csv(os.path.join(fullset_dir('case_2'), 'z=0.8-v=0.3-XY.csv'),
              make_sequence(rng, args.rows, {p: (1500 + 20 * p, 4000) for p in range(4)}))

    print(f"\n合成数据已生成到 {args.out}/fullset/，共 6 个 CSV。")


if __name__ == '__main__':
    main()
