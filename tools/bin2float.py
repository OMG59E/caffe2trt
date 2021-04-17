#!/usr/bin/env python
# -*- coding=utf-8 -*-
"""
@author: xingwg
@license: (C) Copyright 2020-2025.
@contact: xingweiguo@chinasvt.com
@project: algorithms
@file: bin2float.py
@time: 21-4-6 上午10:51
@desc:
"""
import struct
import sys, os
import math


def bin2float(s):
    size = len(s)
    if size % 4 != 0:
        raise Exception('the input bin size should be divisible by 4')
    float_size = int(size/4)
    s = struct.unpack('%df'%float_size, s)
    return s


def get_data(file):
    if not os.path.exists(file):
        raise Exception('%s doesnot exist' % file)
    with open(file, 'r') as f:
        data = f.read()
        if len(data) == 0:
            raise Exception('read %s failed'% file)
        return bin2float(data)


def compare(f0, f1, epsilon, off0=0, off1=0, size=0):
    d0 = get_data(f0)
    d1 = get_data(f1)
    s0 = len(d0)
    s1 = len(d1)

    size = s0

    if s0 != s1:
        print("WARNING:")
        print("  The sizes of the two binaries are not same (", s0, "vs", s1, ")")
        print("  Ignore some more data")

        if s0 > s1:
            size = s1
            d0 = d0[:size]

        if s1 > s0:
            size = s0
            d1 = d1[:size]

    msg = ""
    failed = 0
    for i in range(0, size):
        if abs(d0[i] - d1[i]) > epsilon or math.isnan(d0[i]) or math.isnan(d1[i]):
            if failed == 0:
                msg = "*** Data Error: [%d %d] %f vs %f" % (i, i, d0[i], d1[i])
            failed += 1

    if failed == 0:
        print("*** Similarity: 1.0 (%d/%d)" % (size, size))
    else:
        similarity = float(size - failed) / size
        msg += ", Similarity: %f (%d/%d) " % (similarity, size - failed, size)
        raise Exception(msg)


def dump(f0, dec_places=6):
    s0 = get_data(f0)
    size = len(s0)
    full = int(size / 10)
    remain = int(size % 10)
    _c = 0
    for i in range(0, full):
        _s = s0[_c : _c + 10]
        _s = list(map(lambda x: "%f" % x, _s))
        _c += 10
        print(' '.join(_s))

    if remain != 0:
        _s = list(map(lambda x: "%f" % x, s0[_c:]))
        print(' '.join(_s))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:  ./bin2float.py compare file0.bin file1.bin [epsilon]")
        print("        ./bin2float.py dump file0.bin")
        exit(1)

    epsilon = 0.000001
    try:
        if sys.argv[1] == 'compare':
            f0 = sys.argv[2]
            f1 = sys.argv[3]
            if len(sys.argv) == 5:
                epsilon = float(sys.argv[4])
            print(f0, ' vs ', f1)
            compare(f0, f1, epsilon)
        elif sys.argv[1] == 'dump':
            f0 = sys.argv[2]
            dump(f0)

    except Exception as e:
        print(e.message)
        exit(1)
    exit(0)
