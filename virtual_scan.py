#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
virtual_scan.py — виртуальный лазерный сканер над конвейером.

Зачем: проверить, выживает ли критерий "круг в сечении", если видно только
ВЕРХ товара. На реальной ленте низ закрыт, и лежащий цилиндр даёт дугу,
а не окружность. Если K на дуге просядет, решение сломается на демо.

Что делает:
  1. Кладёт STL на ленту в устойчивую позу (как он реально ляжет).
  2. Снимает сверху карту высот с шагом сканера — получается облако точек
     только верхней поверхности, ровно как у настоящего сканера.
  3. Добавляет шум, выпадение точек (тёмные и блестящие места) и провалы
     на крутых стенках, которые сканер видит по касательной.
  4. Считает K по вертикальным сечениям, замыкая контур плоскостью ленты.
  5. Сравнивает с эталоном по полной сетке.

Запуск: python3 virtual_scan.py -i ./stl
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union

K_THRESHOLD = 0.80
MIN_AREA_FRAC = 0.10
MIN_R_MM = 3.0


# ---------------------------------------------------------------- сканер
def stable_pose(mesh):
    """Как товар реально ляжет на ленту: самая вероятная устойчивая поза."""
    m = mesh.copy()
    try:
        transforms, probs = trimesh.poses.compute_stable_poses(m, n_samples=8)
        if len(transforms):
            m.apply_transform(transforms[int(np.argmax(probs))])
    except Exception:
        pass
    m.apply_translation([0, 0, -m.bounds[0][2]])   # ставим на ленту z=0
    return m


def scan_from_above(mesh, res=0.5, noise=0.3, dropout=0.02, steep_deg=75.0,
                    n_samples=400_000, seed=0):
    """Карта высот сверху -> облако точек только видимой поверхности.

    Вместо трассировки лучей: плотно сэмплим поверхность и в каждой ячейке
    сетки оставляем САМУЮ ВЫСОКУЮ точку. Это и есть вид сверху — низ товара
    отбрасывается сам собой, ровно как у настоящего сканера.
    """
    rng = np.random.default_rng(seed)
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_samples)
    normals = mesh.face_normals[face_idx]

    # крутые стенки сканер видит по касательной — часть точек теряется
    cos_steep = math.cos(math.radians(steep_deg))
    steep = normals[:, 2] < cos_steep
    keep = ~(steep & (rng.random(len(pts)) < 0.6))
    pts = pts[keep]

    # сетка сканера: в каждой ячейке остаётся верхняя точка
    ij = np.floor(pts[:, :2] / res).astype(np.int64)
    key = ij[:, 0].astype(np.int64) * 100003 + ij[:, 1]
    order = np.lexsort((-pts[:, 2], key))          # внутри ячейки z по убыванию
    pts, key = pts[order], key[order]
    first = np.concatenate(([True], key[1:] != key[:-1]))
    cloud = pts[first]

    # шум по высоте и случайные выпадения (тёмное, блестящее, прозрачное)
    cloud = cloud.copy()
    cloud[:, 2] += rng.normal(0, noise, len(cloud))
    cloud = cloud[rng.random(len(cloud)) > dropout]
    return cloud


# ---------------------------------------------------------------- геометрия
def footprint_axes(cloud):
    """Горизонтальные оси товара: минимальный прямоугольник по следу на ленте."""
    xy = cloud[:, :2]
    hull = shapely.convex_hull(shapely.multipoints(xy))
    rect = np.array(shapely.oriented_envelope(hull).exterior.coords)[:4]
    e = rect[1] - rect[0]
    n = np.linalg.norm(e)
    if n < 1e-9:
        return np.eye(2)
    e = e / n
    return np.array([e, [-e[1], e[0]]])            # длинная и короткая оси


def fit_circle(v, z):
    """Подгонка окружности к дуге (алгебраический метод). -> (cx, cz, R, rms)"""
    A = np.column_stack([v, z, np.ones(len(v))])
    b = v ** 2 + z ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cz = sol[0] / 2.0, sol[1] / 2.0
    R2 = sol[2] + cx ** 2 + cz ** 2
    if R2 <= 0:
        return None
    R = math.sqrt(R2)
    d = np.hypot(v - cx, z - cz)
    return cx, cz, R, float(np.sqrt(np.mean((d - R) ** 2)))


def arc_is_circle(v, z):
    """Верхняя дуга — это окружность?

    У тела, лежащего на ленте, сканер видит только верх. Замыкание контура
    лентой раздувает описанную окружность: идеальный круг даёт K = 0.8 —
    ровно порог, худшее место. Поэтому невидимый низ ДОСТРАИВАЕМ: подгоняем
    окружность к видимой дуге. Если она садится хорошо, сечение и есть круг.

    Плоский верх коробки тоже "ложится" на окружность — но огромного радиуса.
    Поэтому требуем, чтобы радиус был сопоставим с полушириной сечения.
    """
    if len(v) < 8:
        return False
    fit = fit_circle(v, z)
    if fit is None:
        return False
    cx, cz, R, rms = fit
    W = float(v.max() - v.min())
    if R < MIN_R_MM or W < 2 * MIN_R_MM:
        return False
    if abs(R - W / 2.0) > 0.30 * (W / 2.0):     # плоский верх -> R огромный
        return False
    if rms > 0.06 * R:                           # дуга не круглая
        return False
    ang = np.unwrap(np.sort(np.arctan2(z - cz, v - cx)))
    if float(ang[-1] - ang[0]) < math.radians(100):   # слишком короткая дуга
        return False
    return True


def k_of_polygon(poly):
    p = poly.simplify(0.2, preserve_topology=True)
    if p.is_empty or p.area <= 0:
        p = poly
    R = float(shapely.minimum_bounding_radius(p))
    if R < MIN_R_MM:
        return None
    seg = shapely.maximum_inscribed_circle(p, tolerance=R * 1e-3)
    return float(seg.length) / R


def height_map(cloud, res=1.0):
    """Растр высот сверху: ровно то, что отдаёт сканер."""
    xy = cloud[:, :2]
    mn = xy.min(axis=0)
    ij = np.floor((xy - mn) / res).astype(np.int64)
    w, h = ij[:, 0].max() + 1, ij[:, 1].max() + 1
    H = np.zeros((h, w), dtype=np.float32)
    np.maximum.at(H, (ij[:, 1], ij[:, 0]), cloud[:, 2].astype(np.float32))
    return H, res


def k_horizontal(cloud, res=1.0, n_levels=20):
    """K по ГОРИЗОНТАЛЬНЫМ сечениям.

    Для выпуклого тела горизонтальное сечение на высоте h — это в точности
    область, где карта высот не ниже h. То есть его видно сверху ТОЧНО,
    без всяких допущений про невидимый низ. Именно так ловится цилиндр,
    стоящий на торце: сверху он круг, а сбоку выглядел бы коробкой.
    """
    H, res = height_map(cloud, res)
    zmax = float(H.max())
    if zmax < MIN_R_MM:
        return 0.0

    best = 0.0
    areas, polys = [], []
    for lvl in np.linspace(0.05 * zmax, 0.95 * zmax, n_levels):
        mask = (H >= lvl).astype(np.uint8)
        # затыкаем дырки от выпавших точек: иначе вписанная окружность честно
        # обходит каждый артефакт и схлопывается в ноль
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.medianBlur(mask, 3)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP,
                                      cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or hier is None:
            continue
        hier = hier[0]
        shells = [(i, c) for i, c in enumerate(cnts)
                  if hier[i][3] < 0 and len(c) >= 4]
        if not shells:
            continue
        i, c = max(shells, key=lambda t: cv2.contourArea(t[1]))
        shell_area = cv2.contourArea(c)
        # настоящие отверстия оставляем, шум от выпавших точек отбрасываем
        holes = [cnts[j] for j in range(len(cnts))
                 if hier[j][3] == i and len(cnts[j]) >= 4
                 and cv2.contourArea(cnts[j]) > 0.02 * shell_area]
        try:
            p = Polygon((c[:, 0, :] * res),
                        [hh[:, 0, :] * res for hh in holes]).buffer(0)
        except Exception:
            continue
        if isinstance(p, Polygon) and p.area > 0:
            polys.append(p); areas.append(p.area)

    if not polys:
        return 0.0
    amax = max(areas)
    for p, a in zip(polys, areas):
        if a < MIN_AREA_FRAC * amax:
            continue
        k = k_of_polygon(p)
        if k and k > best:
            best = k
    return best


def k_from_cloud(cloud, n_slices=25, slab=None, res_v=1.0):
    """K по вертикальным сечениям облака, контур замыкается плоскостью ленты."""
    A = footprint_axes(cloud)
    local = cloud[:, :2] @ A.T                     # (вдоль, поперёк)
    z = cloud[:, 2]

    best = 0.0
    for axis in (0, 1):                            # режем поперёк каждой оси
        u = local[:, axis]                         # координата вдоль реза
        v = local[:, 1 - axis]                     # ось "ширина" в сечении
        lo, hi = u.min(), u.max()
        if hi - lo < 2 * MIN_R_MM:
            continue
        half = (hi - lo) / (2 * n_slices) if slab is None else slab / 2
        centers = np.linspace(lo + 2 * half, hi - 2 * half, n_slices)

        polys = []
        for c in centers:
            m = np.abs(u - c) <= half
            if m.sum() < 8:
                continue
            vv, zz = v[m], z[m]

            # ВЕРХНЯЯ ОГИБАЮЩАЯ: бинуем по ширине и берём максимум по высоте.
            # Без этого точки идут вперемешку, контур сам себя пересекает
            # и полигон вырождается в ноль.
            nb = max(12, int((vv.max() - vv.min()) / max(res_v, 1e-6)))
            edges = np.linspace(vv.min(), vv.max(), nb + 1)
            bi = np.clip(np.digitize(vv, edges) - 1, 0, nb - 1)
            top = np.full(nb, -np.inf)
            np.maximum.at(top, bi, zz)
            ok = np.isfinite(top)
            if ok.sum() < 6:
                continue
            vc = 0.5 * (edges[:-1] + edges[1:])[ok]
            zc = np.maximum(top[ok], 0.0)

            # достроили низ по дуге -> сечение и есть окружность, K = 1
            if arc_is_circle(vc, zc):
                return 1.0

            # иначе — замыкаем контур плоскостью ленты (z = 0)
            ring = np.vstack([
                np.column_stack([vc, zc]),
                [[vc[-1], 0.0], [vc[0], 0.0]],
            ])
            try:
                p = Polygon(ring).buffer(0)
            except Exception:
                continue
            if isinstance(p, Polygon) and p.area > 0:
                polys.append(p)

        if not polys:
            continue
        amax = max(p.area for p in polys)
        for p in polys:
            if p.area < MIN_AREA_FRAC * amax:
                continue
            k = k_of_polygon(p)
            if k and k > best:
                best = k
    return best


# ---------------------------------------------------------------- прогон
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--res", type=float, default=0.5, help="шаг сканера, мм")
    ap.add_argument("--noise", type=float, default=0.3, help="шум по высоте, мм")
    ap.add_argument("--dropout", type=float, default=0.02, help="доля потерянных точек")
    args = ap.parse_args()

    files = sorted(p for p in Path(args.input).glob("*")
                   if p.suffix.lower() in {".stl", ".obj", ".ply"})

    print(f"{'файл':<18}{'точек':>8}{'K верт':>10}{'K гориз':>10}{'K':>8}{'вывод':>14}")
    print("-" * 68)
    for f in files:
        mesh = trimesh.load(str(f), force="mesh")
        mesh = stable_pose(mesh)
        cloud = scan_from_above(mesh, res=args.res, noise=args.noise,
                                dropout=args.dropout)
        kv = k_from_cloud(cloud)
        kh = k_horizontal(cloud, res=max(args.res, 1.0))
        k = max(kv, kh)
        verdict = "круглый" if k > K_THRESHOLD else "не круглый"
        print(f"{f.stem:<18}{len(cloud):>8}{kv:>10.3f}{kh:>10.3f}{k:>8.3f}{verdict:>14}")


if __name__ == "__main__":
    main()
