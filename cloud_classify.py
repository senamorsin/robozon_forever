#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cloud_classify.py — классификация товара по ОБЛАКУ ТОЧЕК с сенсора.

Отличие от stl_classify.py:
  stl_classify — офлайн, вход это идеальная замкнутая сетка, эталон для разметки.
  cloud_classify — рантайм на железе, вход это облако точек с камер/лазера:
  видна только верхняя поверхность, низ закрыт лентой, есть шум и дырки.

Что делает и почему так:
  1. Габариты — по горизонтальному следу на ленте (минимальный прямоугольник)
     плюс высота. Это устойчивее к дыркам, чем OBB по разреженному облаку.
  2. Критерий круга — ДВА типа сечений, потому что один не покрывает всё:
       - горизонтальные: ловят цилиндр, стоящий на торце (сверху виден круг);
       - вертикальные с достройкой дуги: ловят цилиндр/бутылку лёжа.
     Низ достраивается подгонкой окружности к видимой дуге, иначе замыкание
     контура лентой раздувает описанную окружность и K садится на порог.
  3. Категорию выдаёт ОБЩАЯ функция rules.classify — та же, что у STL-версии,
     чтобы рантайм и эталон не разошлись в правилах.
  4. Дополнительно к категории: поза захвата для клешни (ось + ширина) и флаг
     "укатился" (скорость объекта не совпала со скоростью ленты).

Допуски (из уточнения организаторов):
  линейные габариты — 5 мм на сторону; объём — 10 %.

Вход в рантайме — numpy-массив (N, 3) в мм, лента = плоскость z = 0.
Для проверки без железа облако генерируется из STL виртуальным сканером.
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import shapely
from shapely.geometry import Polygon

# ---- единые правила: тянем пороги и решающую функцию из STL-версии ----
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "rules", str(Path(__file__).parent / "stl_classify.py"))
rules = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rules)

# Порог круга по ОБОЛОЧКЕ смещён: измерение снаружи завышает K (см. раздел 11
# отчёта). По ПОВЕРХНОСТИ (лазер) остаётся 0.80. Сенсор сообщает, каким
# каналом получено облако, и мы берём соответствующий порог.
K_HULL = 0.87           # облако из силуэтов (камеры)
K_SURFACE = 0.80        # облако с лазера (истинная поверхность)
MIN_AREA_FRAC = 0.10
MIN_R_MM = 3.0
BELT_SPEED = 1000.0     # мм/с, задано условиями и известно точно


# ======================================================================
# ГЕОМЕТРИЯ
# ======================================================================
def _k_of_polygon(poly):
    p = poly.simplify(0.3, preserve_topology=True)
    if p.is_empty or p.area <= 0:
        p = poly
    R = float(shapely.minimum_bounding_radius(p))
    if R < MIN_R_MM:
        return None
    seg = shapely.maximum_inscribed_circle(p, tolerance=R * 1e-3)
    return float(seg.length) / R


def _footprint(cloud):
    """След на ленте: минимальный прямоугольник, даёт длину/ширину и оси."""
    xy = cloud[:, :2]
    hull = shapely.convex_hull(shapely.multipoints(xy))
    env = shapely.oriented_envelope(hull)
    rect = np.array(env.exterior.coords)[:4]
    e0 = rect[1] - rect[0]; e1 = rect[2] - rect[1]
    L0, L1 = np.linalg.norm(e0), np.linalg.norm(e1)
    long_e = e0 / (L0 + 1e-9) if L0 >= L1 else e1 / (L1 + 1e-9)
    axes = np.array([long_e, [-long_e[1], long_e[0]]])
    return max(L0, L1), min(L0, L1), axes


def dimensions(cloud):
    """Габариты товара: длина, ширина по следу + высота по облаку."""
    L, W, axes = _footprint(cloud)
    H = float(cloud[:, 2].max() - min(cloud[:, 2].min(), 0.0))
    return sorted([L, W, H], reverse=True), axes


def _fit_circle(v, z):
    A = np.column_stack([v, z, np.ones(len(v))])
    b = v ** 2 + z ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    cx, cz = sol[0] / 2, sol[1] / 2
    R2 = sol[2] + cx ** 2 + cz ** 2
    if R2 <= 0:
        return None
    R = math.sqrt(R2)
    d = np.hypot(v - cx, z - cz)
    return cx, cz, R, float(np.sqrt(np.mean((d - R) ** 2)))


def _arc_is_circle(v, z):
    """Верхняя дуга достраивается до окружности? (тело лежит на ленте)"""
    if len(v) < 8:
        return False
    fit = _fit_circle(v, z)
    if fit is None:
        return False
    cx, cz, R, rms = fit
    Wd = float(v.max() - v.min())
    if R < MIN_R_MM or Wd < 2 * MIN_R_MM:
        return False
    if abs(R - Wd / 2) > 0.30 * (Wd / 2):     # плоский верх -> R огромный
        return False
    if rms > 0.06 * R:
        return False
    ang = np.unwrap(np.sort(np.arctan2(z - cz, v - cx)))
    return float(ang[-1] - ang[0]) >= math.radians(100)


def k_vertical(cloud, axes, n_slices=25):
    """K по вертикальным сечениям с достройкой невидимого низа."""
    local = cloud[:, :2] @ axes.T
    z = cloud[:, 2]
    best = 0.0
    for a in (0, 1):
        u, v = local[:, a], local[:, 1 - a]
        lo, hi = u.min(), u.max()
        if hi - lo < 2 * MIN_R_MM:
            continue
        half = (hi - lo) / (2 * n_slices)
        polys = []
        for c in np.linspace(lo + 2 * half, hi - 2 * half, n_slices):
            m = np.abs(u - c) <= half
            if m.sum() < 8:
                continue
            vv, zz = v[m], z[m]
            nb = max(12, int((vv.max() - vv.min())))
            edges = np.linspace(vv.min(), vv.max(), nb + 1)
            bi = np.clip(np.digitize(vv, edges) - 1, 0, nb - 1)
            top = np.full(nb, -np.inf)
            np.maximum.at(top, bi, zz)
            ok = np.isfinite(top)
            if ok.sum() < 6:
                continue
            vc = (0.5 * (edges[:-1] + edges[1:]))[ok]
            zc = np.maximum(top[ok], 0.0)
            if _arc_is_circle(vc, zc):
                return 1.0
            ring = np.vstack([np.column_stack([vc, zc]),
                              [[vc[-1], 0.0], [vc[0], 0.0]]])
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
            k = _k_of_polygon(p)
            if k and k > best:
                best = k
    return best


def k_horizontal(cloud, res=1.0, n_levels=20):
    """K по горизонтальным сечениям (ловит цилиндр на торце)."""
    xy = cloud[:, :2]; mn = xy.min(axis=0)
    ij = np.floor((xy - mn) / res).astype(np.int64)
    w, h = ij[:, 0].max() + 1, ij[:, 1].max() + 1
    Hm = np.zeros((h, w), np.float32)
    np.maximum.at(Hm, (ij[:, 1], ij[:, 0]), cloud[:, 2].astype(np.float32))
    zmax = float(Hm.max())
    if zmax < MIN_R_MM:
        return 0.0
    best = 0.0; polys = []; areas = []
    for lvl in np.linspace(0.05 * zmax, 0.95 * zmax, n_levels):
        mask = (Hm >= lvl).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.medianBlur(mask, 3)
        cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts or hier is None:
            continue
        hier = hier[0]
        shells = [(i, c) for i, c in enumerate(cnts) if hier[i][3] < 0 and len(c) >= 4]
        if not shells:
            continue
        i, c = max(shells, key=lambda t: cv2.contourArea(t[1]))
        sa = cv2.contourArea(c)
        holes = [cnts[j][:, 0, :] * res for j in range(len(cnts))
                 if hier[j][3] == i and len(cnts[j]) >= 4
                 and cv2.contourArea(cnts[j]) > 0.02 * sa]
        try:
            p = Polygon(c[:, 0, :] * res, holes).buffer(0)
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
        k = _k_of_polygon(p)
        if k and k > best:
            best = k
    return best


# ======================================================================
# ПОЗА ЗАХВАТА ДЛЯ КЛЕШНИ
# ======================================================================
def grasp_for_gripper(cloud, dims, axes):
    """Для клешни нужны ось захвата и ширина раскрытия, а НЕ точка присоски.

    Круглый товар берётся поперёк оси (за диаметр). Ось цилиндра — это длинная
    ось следа на ленте; ширина раскрытия — короткий габарит плюс запас.
    Центр захвата — центр следа. Всё это уже посчитано для критерия круга,
    поза выпадает почти бесплатно.
    """
    cx, cy = cloud[:, 0].mean(), cloud[:, 1].mean()
    long_axis = axes[0]                       # вдоль длинной стороны
    width = dims[1]                           # короткий из (L, W)
    return {
        "center_xy": [round(float(cx), 1), round(float(cy), 1)],
        "grip_axis": [round(float(long_axis[0]), 3), round(float(long_axis[1]), 3)],
        "open_mm": round(float(width) + 10.0, 1),   # +10 мм запас на раскрытие
        "top_z": round(float(cloud[:, 2].max()), 1),
    }


# ======================================================================
# УКАТЫВАНИЕ: скорость объекта против скорости ленты
# ======================================================================
def rolling_flag(track):
    """Определить, катится ли товар (краевой случай от организаторов).

    track: список (t_сек, x_центра_мм) по кадрам одного объекта.
    Едет с лентой -> dx/dt ≈ BELT_SPEED. Катится -> заметно отличается
    (быстрее при скатывании вперёд, медленнее/вбок при застревании).
    """
    if track is None or len(track) < 2:
        return {"rolling": False, "reason": "мало кадров"}
    t = np.array([p[0] for p in track], float)
    x = np.array([p[1] for p in track], float)
    v = np.polyfit(t, x, 1)[0]                # скорость центра, мм/с
    dev = abs(v - BELT_SPEED) / BELT_SPEED
    return {
        "rolling": bool(dev > 0.15),          # >15 % расхождения = катится
        "object_speed": round(float(v), 1),
        "belt_speed": BELT_SPEED,
        "deviation": round(float(dev), 3),
    }


# ======================================================================
# ГЛАВНАЯ ФУНКЦИЯ РАНТАЙМА
# ======================================================================
def classify_cloud(cloud, source="hull", track=None,
                   size_tol=5.0, vol_tol=0.10):
    """Полный результат по одному товару. Готов к отправке контроллеру.

    source: "hull" (камеры) | "surface" (лазер) — задаёт порог круга.
    track:  опционально, кадры позиции для детектора укатывания.
    """
    cloud = np.asarray(cloud, float)
    if len(cloud) < 30:
        return {"category": "error", "zone": "R", "reason": "мало точек"}

    dims, axes = dimensions(cloud)

    # критерий круга: оба типа сечений, берём максимум
    kv = k_vertical(cloud, axes)
    kh = k_horizontal(cloud)
    k = max(kv, kh)

    # порог зависит от источника; подменяем в общих правилах на время вызова
    k_thr = K_SURFACE if source == "surface" else K_HULL
    saved = rules.K_THRESHOLD, rules.SIZE_TOL
    rules.K_THRESHOLD = k_thr
    rules.SIZE_TOL = size_tol
    try:
        cat, reason, b_size, b_round = rules.classify(dims, k)
    finally:
        rules.K_THRESHOLD, rules.SIZE_TOL = saved

    zone = {rules.CAT_OK: "B", rules.CAT_SIZE: "C", rules.CAT_ROUND: "D"}[cat]

    out = {
        "category": cat,
        "zone": zone,
        "reason": reason,
        "dims_mm": [round(d, 1) for d in dims],
        "k_max": round(k, 3),
        "k_vertical": round(kv, 3),
        "k_horizontal": round(kh, 3),
        "k_threshold": k_thr,
        "source": source,
        "borderline_size": bool(b_size),
        "borderline_round": bool(b_round),
        "npoints": int(len(cloud)),
    }
    # круглым нужна поза для клешни
    if cat == rules.CAT_ROUND:
        out["grasp"] = grasp_for_gripper(cloud, dims, axes)
    # укатывание
    if track is not None:
        roll = rolling_flag(track)
        out["roll"] = roll
        if roll["rolling"]:
            out["zone"] = "R"                 # укатился -> разбор/повторный прогон
            out["reason"] = "объект катится, скорость не совпала с лентой"
    return out


# ======================================================================
# ПРОВЕРКА БЕЗ ЖЕЛЕЗА: облако из STL виртуальным сканером
# ======================================================================
def _demo(input_dir, source):
    import trimesh
    vs_spec = importlib.util.spec_from_file_location(
        "vs", str(Path(__file__).parent / "virtual_scan.py"))
    vs = importlib.util.module_from_spec(vs_spec); vs_spec.loader.exec_module(vs)

    files = sorted(p for p in Path(input_dir).glob("*")
                   if p.suffix.lower() in {".stl", ".obj", ".ply"})
    print(f"источник: {source}   допуск: 5 мм / 10 % объёма\n")
    print(f"{'файл':<16}{'зона':>6}{'габариты':>22}{'K':>8}  причина")
    print("-" * 78)
    for f in files:
        mesh = vs.stable_pose(trimesh.load(str(f), force="mesh"))
        cloud = vs.scan_from_above(mesh, res=0.5, noise=0.3, dropout=0.02)
        r = classify_cloud(cloud, source=source)
        d = "×".join(f"{x:.0f}" for x in r["dims_mm"])
        print(f"{f.stem:<16}{r['zone']:>6}{d:>22}{r['k_max']:>8.3f}  {r['reason']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True, help="папка со STL для проверки")
    ap.add_argument("--source", choices=["hull", "surface"], default="hull",
                    help="hull=камеры (порог 0.87), surface=лазер (порог 0.80)")
    args = ap.parse_args()
    _demo(args.input, args.source)
