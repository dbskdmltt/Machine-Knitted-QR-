#!/usr/bin/env python3
"""Integer-mapped knitting chart for a QR code (1 module = SX stitches x SY courses).

python make_chart.py --url https://zrr.kr/ZFyyPM --version 3 --ecc H --sx 2 --sy 3 --border 4
-> chart_as_seen.png (face side) and chart_mirrored_for_machine.png (input for a single-bed machine)
"""
import argparse, numpy as np, qrcode
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('--url', default='https://zrr.kr/ZFyyPM')
ap.add_argument('--version', type=int, default=3)
ap.add_argument('--ecc', default='H', choices='LMQH')
ap.add_argument('--sx', type=int, default=2)
ap.add_argument('--sy', type=int, default=3)
ap.add_argument('--border', type=int, default=4)
a = ap.parse_args()
ecc = dict(L=qrcode.constants.ERROR_CORRECT_L, M=qrcode.constants.ERROR_CORRECT_M,
           Q=qrcode.constants.ERROR_CORRECT_Q, H=qrcode.constants.ERROR_CORRECT_H)[a.ecc]
q = qrcode.QRCode(version=a.version, error_correction=ecc, border=a.border)
q.add_data(a.url); q.make(fit=False)
M = np.array(q.get_matrix(), np.uint8)              # 1 = dark module, border included
chart = np.kron(M, np.ones((a.sy, a.sx), np.uint8))  # rows = courses, cols = stitches
print('chart', chart.shape[1], 'stitches x', chart.shape[0], 'courses')
Image.fromarray(255 * (1 - chart)).save('chart_as_seen.png')
Image.fromarray(255 * (1 - chart[:, ::-1])).save('chart_mirrored_for_machine.png')
