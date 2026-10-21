#!/usr/bin/env python3
"""
knitqr_analyze.py — 편직 QR 샘플 사진 정량 분석 파이프라인

사진 폴더를 넣으면 샘플별로 다음을 계산해 CSV로 저장한다.
  - 모듈 오류율 (설계 도안 대비, 국소 정합 후)
  - 전역 정합(원근 보정)만 했을 때의 모듈 오류율
  - 비투영 변형량(천의 휨): 원근 보정으로 설명되지 않는 잔차 변위 RMS / 최대 (모듈 단위)
  - 원본 사진 직접 디코딩 성공 여부 (zbar, OpenCV)
  - 좌우반전 여부

사용 예:
  python knitqr_analyze.py photos/ --chart QR_knit_chart_74x111_as_seen.png --sx 2 --sy 3 --border 4
  python knitqr_analyze.py photos/ --url https://zrr.kr/ZFyyPM --version 3 --ecc H
"""
import argparse, csv, os, sys, glob, itertools
import numpy as np
import cv2

try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol
    QR_ONLY = [ZBarSymbol.QRCODE]
except Exception:  # zbar 라이브러리가 없으면 OpenCV 디코더만 사용
    pyzbar = None

P = 16      # 정규화 좌표계에서 모듈 1칸의 픽셀 크기
B = 4       # 정규화 좌표계의 여백(모듈)


# ---------------------------------------------------------------- 설계 도안
def truth_from_chart(path, sx, sy, border):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"도안 파일을 열 수 없음: {path}")
    rows, cols = img.shape
    nr, nc = rows // sy, cols // sx
    M = np.zeros((nr, nc), int)
    for r in range(nr):
        for c in range(nc):
            M[r, c] = 1 if img[r * sy + sy // 2, c * sx + sx // 2] < 128 else 0
    T = M[border:nr - border, border:nc - border]
    if T.shape[0] != T.shape[1]:
        sys.exit(f"도안에서 읽은 모듈 행렬이 정사각형이 아님: {T.shape}. --sx/--sy/--border 확인")
    return T


def truth_from_url(url, version, ecc):
    import qrcode
    lv = {"L": qrcode.constants.ERROR_CORRECT_L, "M": qrcode.constants.ERROR_CORRECT_M,
          "Q": qrcode.constants.ERROR_CORRECT_Q, "H": qrcode.constants.ERROR_CORRECT_H}[ecc]
    q = qrcode.QRCode(version=version, error_correction=lv, border=0)
    q.add_data(url)
    q.make(fit=version is None)
    return np.array(q.get_matrix(), int)


def alignment_center(N):
    """버전 2 이상에서 오른쪽 아래 정렬 패턴의 모듈 중심 좌표."""
    return None if N <= 21 else N - 7 + 0.5


# ---------------------------------------------------------------- 파인더 검출
def normalize(gray):
    g = gray.astype(np.float32)
    k = max(31, (min(g.shape) // 8) | 1)
    return g / (cv2.blur(g, (k, k)) + 1e-3)


def finder_score(dark, cx, cy, m):
    """중심 (cx,cy), 모듈 크기 m 에서 1:1:3:1:1 링 구조를 8방향으로 검사."""
    h, w = dark.shape
    expect = [(0.0, 1), (1.0, 1), (2.0, 0), (3.0, 1), (4.0, 0)]   # (중심거리/모듈, 어두움)
    ok = tot = 0
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1), (.707, .707), (-.707, .707), (.707, -.707), (-.707, -.707)]:
        diag = abs(dx) < 1 and abs(dy) < 1
        for d, want in expect:
            if diag and d == 4.0:
                continue
            dd = d * (1.414 if diag else 1.0)
            x, y = int(cx + dx * dd * m), int(cy + dy * dd * m)
            r = max(1, int(m * 0.25))
            if not (r <= x < w - r and r <= y < h - r):
                tot += 1
                continue
            v = dark[y - r:y + r + 1, x - r:x + r + 1].mean() > 0.5
            ok += (v == bool(want))
            tot += 1
    return ok / tot


def find_finders(gray, N):
    gn = normalize(gray)
    gb = cv2.GaussianBlur(gn, (5, 5), 0)
    thr = cv2.threshold((np.clip(gb, 0, 2) * 127).astype(np.uint8), 0, 255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    dark = (thr > 0).astype(np.float32)
    H, W = gray.shape
    cands = []
    ks = np.unique(np.geomspace(min(H, W) * 0.012, min(H, W) * 0.12, 14).astype(int))
    for k in ks:
        if k < 3:
            continue
        o = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        cs, _ = cv2.findContours(o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cn in cs:
            x, y, w, h = cv2.boundingRect(cn)
            if not (k <= w <= 2.2 * k and k <= h <= 2.2 * k):
                continue
            if not (0.6 < w / h < 1.65):
                continue
            cx, cy = x + w / 2, y + h / 2
            m = (w + h) / 2 / 3.0
            s = finder_score(dark, cx, cy, m)
            if s >= 0.72:
                cands.append((s, cx, cy, m))
    # 중복 제거
    cands.sort(reverse=True)
    uniq = []
    for c in cands:
        if all(np.hypot(c[1] - u[1], c[2] - u[2]) > 2 * c[3] for u in uniq):
            uniq.append(c)
    uniq = uniq[:12]
    best = None
    for tri in itertools.combinations(uniq, 3):
        ms = [t[3] for t in tri]
        if max(ms) / min(ms) > 1.7:
            continue
        m = np.median(ms)
        pts = [np.array([t[1], t[2]]) for t in tri]
        for i in range(3):   # i = 직각 꼭짓점(왼쪽 위 파인더)
            a, b = [j for j in range(3) if j != i]
            va, vb = pts[a] - pts[i], pts[b] - pts[i]
            la, lb = np.linalg.norm(va), np.linalg.norm(vb)
            cos = abs(va @ vb) / (la * lb)
            if cos > 0.25 or abs(la - lb) / max(la, lb) > 0.3:
                continue
            exp = (N - 7) * m
            scale_err = abs((la + lb) / 2 - exp) / exp
            if scale_err > 0.45:
                continue
            score = sum(t[0] for t in tri) - cos - abs(la - lb) / max(la, lb) - scale_err
            if best is None or score > best[0]:
                # 영상 좌표(y 아래 방향)에서 외적 > 0 이 되도록 TR/BL 지정
                cross = va[0] * vb[1] - va[1] * vb[0]
                tr, bl = (pts[a], pts[b]) if cross > 0 else (pts[b], pts[a])
                best = (score, pts[i], tr, bl, m)
    return best, gn


# ---------------------------------------------------------------- 정합과 샘플링
def design_image(T):
    return np.kron(1 - np.pad(T, B), np.ones((P, P))).astype(np.float32)


def lattice(N):
    idx = list(range(B + 1, B + N, 3))
    if idx[-1] != B + N - 1:
        idx.append(B + N - 1)
    return idx


def block_match(w, d, N):
    W = w.shape[0]
    R, half = int(1.5 * P), 3 * P
    src, dst, ncc = [], [], []
    for my in lattice(N):
        for mx in lattice(N):
            cy, cx = my * P, mx * P
            y0, x0 = max(cy - half, 0), max(cx - half, 0)
            tmpl = d[y0:cy + half, x0:cx + half]
            if tmpl.std() < 1e-3:
                continue
            ys, xs = max(y0 - R, 0), max(x0 - R, 0)
            reg = w[ys:min(y0 + tmpl.shape[0] + R, W), xs:min(x0 + tmpl.shape[1] + R, W)]
            if reg.shape[0] < tmpl.shape[0] or reg.shape[1] < tmpl.shape[1]:
                continue
            res = cv2.matchTemplate(reg, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mv, _, ml = cv2.minMaxLoc(res)
            dx, dy = ml[0] + xs - x0, ml[1] + ys - y0
            src.append((cx + dx, cy + dy))
            dst.append((cx, cy))
            ncc.append(mv)
    return np.float32(src), np.float32(dst), np.array(ncc)


def displacement_remap(w, src, dst):
    W = w.shape[0]
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32)
    nx = np.zeros_like(xx); ny = np.zeros_like(xx); den = np.zeros_like(xx)
    for (sx, sy), (px, py) in zip(src, dst):
        wg = 1 / ((xx - px) ** 2 + (yy - py) ** 2 + P * P) ** 1.5
        nx += wg * (sx - px); ny += wg * (sy - py); den += wg
    return cv2.remap(w, xx + nx / den, yy + ny / den, cv2.INTER_LINEAR, borderValue=1.0)


def sample(img, N):
    G = np.zeros((N, N), int)
    for r in range(N):
        for c in range(N):
            y, x = (r + B) * P + P // 2, (c + B) * P + P // 2
            G[r, c] = 1 if img[y - 3:y + 4, x - 3:x + 4].mean() < 0.75 else 0
    return G


def decode_grid(G):
    img = np.kron(1 - G, np.ones((10, 10))).astype(np.uint8) * 255
    img = cv2.copyMakeBorder(img, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    out = ""
    if pyzbar:
        d = pyzbar.decode(img, symbols=QR_ONLY)
        if d:
            out = d[0].data.decode("utf-8", "replace")
    if not out:
        out = cv2.QRCodeDetector().detectAndDecode(img)[0]
    return out


def raw_decode(gray):
    """직접 디코딩: 해상도 8단계 × 블러 3단계 = 24회 시도 중 성공 횟수."""
    n = ok = 0
    text = ""
    long_side = max(gray.shape)
    for target in [200, 240, 280, 330, 400, 500, 650, 850]:
        s = min(1.0, target / long_side)
        t0 = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        for k in [1, 3, 5]:
            t = cv2.GaussianBlur(t0, (k, k), 0) if k > 1 else t0
            n += 1
            got = ""
            if pyzbar:
                d = pyzbar.decode(t, symbols=QR_ONLY)
                if d:
                    got = d[0].data.decode("utf-8", "replace")
            if not got:
                got = cv2.QRCodeDetector().detectAndDecode(t)[0]
            if got:
                ok += 1
                text = got
    return ok, n, text


def analyze(path, T, outdir):
    rec = {"file": os.path.basename(path)}
    img = cv2.imread(path)
    if img is None:
        rec["note"] = "이미지 열기 실패"
        return rec
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = min(1.0, 1600 / max(img.shape[:2]))
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    N = T.shape[0]

    ok, n, text = raw_decode(gray_full)
    rec.update(raw_decode_success=f"{ok}/{n}", raw_decoded=text)

    best, gn = find_finders(gray, N)
    if best is None:
        rec["note"] = "파인더 3개 검출 실패"
        return rec
    _, TL, TR, BL, m = best
    rec["module_px"] = round(float(m) / s, 1)
    # 폰으로 QR을 겨냥한 상황을 흉내: 여백 4모듈 포함 영역만 잘라서 디코딩
    ex = (TR - TL) / (N - 7); ey = (BL - TL) / (N - 7)
    o = TL - 3.5 * ex - 3.5 * ey
    corners = np.array([o + a * ex + b * ey for a in (-4, N + 4) for b in (-4, N + 4)])
    x0, y0 = np.maximum((corners.min(0) / s).astype(int), 0)
    x1, y1 = (corners.max(0) / s).astype(int)
    ok2, n2, text2 = raw_decode(gray_full[y0:y1, x0:x1])
    rec.update(crop_decode_success=f"{ok2}/{n2}", raw_decoded=text or text2)

    Wc = (N + 2 * B) * P
    d_true = design_image(T)
    # 1단계: 파인더 3점 아핀 → 국소 정합 → 호모그래피 재추정
    S3 = np.float32([TL, TR, BL])
    D3 = (np.float32([[3.5, 3.5], [N - 3.5, 3.5], [3.5, N - 3.5]]) + B) * P
    Hm = np.vstack([cv2.getAffineTransform(S3, D3), [0, 0, 1]])

    results = {}
    for mirrored in (False, True):
        Tm = T.T if mirrored else T   # 좌우반전 편직물 = 라벨링 후 전치 행렬
        d = design_image(Tm)
        H = Hm.copy()
        for _ in range(2):
            w = cv2.warpPerspective(gn, H, (Wc, Wc), borderValue=1.0)
            src, dst, ncc = block_match(w, d, N)
            if len(src) < 6:
                break
            Hc, _ = cv2.findHomography(src, dst, cv2.RANSAC, P * 0.75)
            if Hc is None:
                break
            H = Hc @ H
        w = cv2.warpPerspective(gn, H, (Wc, Wc), borderValue=1.0)
        G_glob = sample(w, N)
        src, dst, ncc = block_match(w, d, N)
        reg = displacement_remap(w, src, dst)
        G = sample(reg, N)
        err = (G != Tm)
        results[mirrored] = dict(G=G, err=err, glob_err=(G_glob != Tm).sum(), w=w, reg=reg,
                                 src=src, dst=dst, ncc=ncc, Tm=Tm)
    mir = results[True]["err"].sum() < results[False]["err"].sum()
    R = results[mir]
    G, err, Tm = R["G"], R["err"], R["Tm"]

    # 비투영 잔차 변위
    Hr, _ = cv2.findHomography(R["src"], R["dst"], 0)
    pr = cv2.perspectiveTransform(R["src"][None], Hr)[0]
    resid = np.linalg.norm(pr - R["dst"], axis=1) / P
    rec["_lat"] = (R["dst"].copy(), pr.copy(), bool(mir))

    tot = N * N
    grid_text = decode_grid(G.T if mir else G)
    rec.update(
        mirrored=bool(mir),
        modules=tot,
        module_err_pct=round(float(100 * err.sum() / tot), 2),
        missing_dark=int((err & (Tm == 1)).sum()),
        extra_dark=int((err & (Tm == 0)).sum()),
        global_only_err_pct=round(float(100 * R["glob_err"] / tot), 2),
        resid_rms_mod=round(float(np.sqrt((resid ** 2).mean())), 3),
        resid_max_mod=round(float(resid.max()), 3),
        match_ncc_mean=round(float(R["ncc"].mean()), 3),
        grid_decoded=grid_text,
    )
    if rec["match_ncc_mean"] < 0.5:
        rec["note"] = "정합 신뢰도 낮음 — 결과 확인 필요"

    # 시각화: 정합된 사진 + 격자 + 오류 표시 | 변위 벡터
    os.makedirs(outdir, exist_ok=True)
    vis = cv2.cvtColor(np.clip(R["reg"] * 180, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i in range(B * P, (N + B) * P + 1, P):
        cv2.line(vis, (i, B * P), (i, (N + B) * P), (200, 200, 200), 1)
        cv2.line(vis, (B * P, i), ((N + B) * P, i), (200, 200, 200), 1)
    for r in range(N):
        for c in range(N):
            if err[r, c]:
                x0, y0 = (c + B) * P, (r + B) * P
                cv2.rectangle(vis, (x0 + 2, y0 + 2), (x0 + P - 2, y0 + P - 2), (0, 0, 255), 2)
    q = cv2.cvtColor(np.clip(R["w"] * 180, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    q = (q * 0.5 + 127).astype(np.uint8)
    # 비투영 잔차 벡터(원근 보정으로 설명되지 않는 천의 휨), 3배 확대 표시
    # 초록: 0.5모듈 미만(스캐너 허용 범위 내), 빨강: 0.5모듈 이상
    for (qx, qy), (px, py), rr in zip(pr, R["dst"], resid):
        col = (0, 160, 0) if rr < 0.5 else (0, 0, 255)
        cv2.circle(q, (int(px), int(py)), 3, col, -1)
        cv2.arrowedLine(q, (int(px), int(py)), (int(px + (qx - px) * 3), int(py + (qy - py) * 3)),
                        col, 2, tipLength=0.25)
    txt = f"err {rec['module_err_pct']}%  resid RMS {rec['resid_rms_mod']} / max {rec['resid_max_mod']} mod"
    panel = np.hstack([vis, np.full((Wc, 10, 3), 255, np.uint8), q])
    panel = cv2.copyMakeBorder(panel, 36, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(panel, txt, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(outdir, os.path.splitext(rec["file"])[0] + "_analysis.png"), panel)
    return rec


FIELDS = ["file", "raw_decode_success", "crop_decode_success", "raw_decoded", "mirrored", "modules", "module_err_pct",
          "missing_dark", "extra_dark", "global_only_err_pct", "resid_rms_mod", "resid_max_mod",
          "match_ncc_mean", "grid_decoded", "module_px", "note"]


def main():
    ap = argparse.ArgumentParser(description="편직 QR 사진 정량 분석")
    ap.add_argument("input", help="사진 폴더 또는 사진 파일")
    ap.add_argument("--chart", help="겉면 기준(as seen) 편직 도안 PNG (1px = 1코)")
    ap.add_argument("--sx", type=int, default=2, help="모듈당 코 수")
    ap.add_argument("--sy", type=int, default=3, help="모듈당 단 수")
    ap.add_argument("--border", type=int, default=4, help="도안 여백(모듈)")
    ap.add_argument("--url", help="도안 대신 URL로 정답 행렬 생성")
    ap.add_argument("--version", type=int, default=3)
    ap.add_argument("--ecc", default="H", choices=list("LMQH"))
    ap.add_argument("--out", default="knitqr_results")
    a = ap.parse_args()

    if a.chart:
        T = truth_from_chart(a.chart, a.sx, a.sy, a.border)
    elif a.url:
        T = truth_from_url(a.url, a.version, a.ecc)
    else:
        sys.exit("--chart 또는 --url 중 하나가 필요함")
    print(f"정답 행렬: {T.shape[0]}×{T.shape[0]} 모듈")
    if pyzbar is None:
        print("주의: pyzbar/zbar 없음 — OpenCV 디코더만 사용")

    if os.path.isdir(a.input):
        files = sorted(f for f in glob.glob(os.path.join(a.input, "*"))
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".bmp", ".tif", ".tiff")))
    else:
        files = [a.input]
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for f in files:
        print(f"분석 중: {os.path.basename(f)}", flush=True)
        try:
            r = analyze(f, T, os.path.join(a.out, "figures"))
        except Exception as e:
            r = {"file": os.path.basename(f), "note": f"오류: {e}"}
        rows.append(r)
        print("   ", {k: r.get(k) for k in ["raw_decode_success", "crop_decode_success", "mirrored", "module_err_pct",
                                          "resid_rms_mod", "resid_max_mod", "note"] if k in r})
    out_csv = os.path.join(a.out, "results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
    print(f"\n저장: {out_csv}")
    summarize(rows, os.path.join(a.out, "summary_by_condition.csv"))


def summarize(rows, path):
    """파일명이 <샘플>_<조건>_<촬영번호>.jpg 형식이면 조건별 평균·표준편차 요약."""
    groups = {}
    for r in rows:
        parts = os.path.splitext(r["file"])[0].split("_")
        if len(parts) < 3 or "module_err_pct" not in r:
            continue
        groups.setdefault(parts[1], []).append(r)
    if not groups:
        return
    keys = ["module_err_pct", "global_only_err_pct", "resid_rms_mod", "resid_max_mod"]
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.writer(fh)
        wr.writerow(["condition", "n", "samples"] + [f"{k}_{s}" for k in keys for s in ("mean", "sd")]
                    + ["crop_decode_rate", "grid_decoded_rate"])
        for cond, rs in sorted(groups.items()):
            row = [cond, len(rs), len({os.path.splitext(r["file"])[0].split("_")[0] for r in rs})]
            for k in keys:
                v = np.array([float(r[k]) for r in rs])
                row += [round(v.mean(), 3), round(v.std(ddof=1), 3) if len(v) > 1 else ""]
            ok = sum(int(r["crop_decode_success"].split("/")[0]) > 0 for r in rs)
            gd = sum(bool(r.get("grid_decoded")) for r in rs)
            row += [round(ok / len(rs), 3), round(gd / len(rs), 3)]
            wr.writerow(row)
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
