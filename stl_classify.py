#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stl_classify.py — разметка тестового набора STL по правилам трека 3.

Что делает:
  1. Читает все STL (и OBJ/PLY/STEP-конвертированные меши) из папки.
  2. Считает собственные габариты товара — по ориентированному ограничивающему
     ящику (OBB), а не по осям сцены. Иначе куб, лежащий под 45°, "вырастет".
  3. Строит сечения ПЕРПЕНДИКУЛЯРНО собственным осям товара, в каждом считает
     K = r_вписанной / R_описанной. Товар считается "круглым в сечении",
     если хотя бы в одном валидном сечении K > 0.8.
  4. Применяет приоритет правил: сначала габариты, потом форма.
  5. Пишет results.csv и results.json — готовую разметку для обучения и отчёта.

Почему сечения только по собственным осям:
  косое сечение куба даёт правильный шестиугольник с K = cos(30°) = 0.866 > 0.8,
  то есть при произвольных плоскостях "круглым" оказывается любой куб.

Почему отбрасываются вырожденные сечения:
  срез у самого кончика любого гладкого тела стремится к кругу (K → 1).
  Отбрасываем сечения с площадью меньше MIN_AREA_FRAC от максимальной по оси
  и с описанной окружностью меньше MIN_R_MM.

Запуск:
  python3 stl_classify.py --input ./stl --output ./out
  python3 stl_classify.py --input ./stl --output ./out --step 2.0 --scale 1.0
"""

import argparse
import concurrent.futures as cf
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import warnings

import numpy as np
import trimesh

warnings.filterwarnings("ignore", category=DeprecationWarning)
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

# shapely 2.x умеет считать обе окружности на C — это в десятки раз быстрее
# чистого Python. Если версия старая, работают запасные реализации ниже.
_FAST_R = hasattr(shapely, "minimum_bounding_radius")
_FAST_r = hasattr(shapely, "maximum_inscribed_circle")

# ----------------------------------------------------------------------------
# ПАРАМЕТРЫ ЗАДАЧИ (все размеры в миллиметрах)
# ----------------------------------------------------------------------------
MIN_DIM = 10.0                     # минимально допустимый габарит: 10 x 10 x 10
MAX_BOX = (450.0, 320.0, 320.0)    # максимально допустимый габарит сортировщика
K_THRESHOLD = 0.80                 # порог "круга в сечении"

# Серые зоны: погрешность измерения на реальном стенде ±2..3 мм.
# Цена ошибки несимметрична: негабарит в сортировщике = затор,
# годный товар в негабарите = секунды ручного труда. Соотношение ~1:100.
# Поэтому спорное решается консервативно — из потока, а не в сортировщик.
SIZE_TOL = 3.0                     # ±3 мм вокруг любого порога габаритов
K_TOL = 0.02                       # ±0.02 вокруг порога круглости

# Фильтр вырожденных сечений
MIN_AREA_FRAC = 0.10               # площадь среза < 10% от максимальной по оси
MIN_R_MM = 3.0                     # описанная окружность меньше 3 мм — мусор

# Категории (совпадают с формулировками ТЗ)
CAT_OK = "sortable"                # "Подходит для сортировки"        -> зона B
CAT_SIZE = "no_fit_size"           # "Не подходит по габаритам"       -> зона C
CAT_ROUND = "needs_repack"         # "Не подходит без доупаковки"     -> зона D

MESH_EXT = {".stl", ".obj", ".ply", ".off", ".glb", ".gltf", ".3mf"}


# ----------------------------------------------------------------------------
# ГЕОМЕТРИЯ СЕЧЕНИЯ
# ----------------------------------------------------------------------------
def min_enclosing_circle(points):
    """Наименьшая описанная окружность (алгоритм Вельцля).
    points: (N, 2) numpy array. Возвращает (cx, cy, R)."""
    pts = [tuple(p) for p in np.asarray(points, dtype=float)]
    pts = list(dict.fromkeys(pts))          # убираем дубликаты
    random.Random(0).shuffle(pts)           # детерминированный shuffle

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def circle_two(a, b):
        cx, cy = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
        return (cx, cy, dist(a, b) / 2.0)

    def circle_three(a, b, c):
        ax, ay = a; bx, by = b; cx, cy = c
        d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            return None
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay)
              + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx)
              + (cx**2 + cy**2) * (bx - ax)) / d
        return (ux, uy, dist((ux, uy), a))

    def inside(c, p):
        return c is not None and dist((c[0], c[1]), p) <= c[2] + 1e-9

    c = None
    for i, p in enumerate(pts):
        if inside(c, p):
            continue
        c = (p[0], p[1], 0.0)
        for j in range(i):
            q = pts[j]
            if inside(c, q):
                continue
            c = circle_two(p, q)
            for k in range(j):
                r = pts[k]
                if inside(c, r):
                    continue
                cc = circle_three(p, q, r)
                if cc is not None:
                    c = cc
    return c if c else (0.0, 0.0, 0.0)


def max_inscribed_radius(poly: Polygon, r_hi: float, iters: int = 24) -> float:
    """Радиус наибольшей вписанной окружности.
    Бинарный поиск по r: если отрицательный буфер не пуст — окружность влезает.
    Отверстия в полигоне учитываются автоматически."""
    lo, hi = 0.0, r_hi
    for _ in range(iters):
        mid = (lo + hi) / 2.0
        if mid <= 1e-9:
            break
        shrunk = poly.buffer(-mid)
        if shrunk.is_empty or shrunk.area <= 0:
            hi = mid
        else:
            lo = mid
    return lo


def _polys_from_segments(seg2d):
    """Полигоны из массива 2D-отрезков (n, 2, 2) — без Python-цикла."""
    if seg2d is None or len(seg2d) == 0:
        return []
    seg = np.asarray(seg2d, dtype=float)
    live = np.any(np.abs(seg[:, 0] - seg[:, 1]) > 1e-9, axis=1)
    seg = seg[live]
    if len(seg) == 0:
        return []

    coords = np.round(seg.reshape(-1, 2), 6)
    idx = np.repeat(np.arange(len(seg)), 2)
    lines = shapely.linestrings(coords, indices=idx)     # пакетно, на C

    merged = shapely.union_all(lines)
    polys = [p for p in polygonize(merged) if p.is_valid and p.area > 0]

    if not polys:
        # контур не замкнулся (дырявая сетка) — заращиваем микробуфером
        try:
            grown = merged.buffer(1e-3).buffer(-1e-3)
            cand = getattr(grown, "geoms", [grown])
            polys = [p for p in cand
                     if isinstance(p, Polygon) and p.is_valid and p.area > 0]
        except Exception:
            return []
    if not polys:
        return []

    # вычитаем вложенные контуры: polygonize отдаёт дырки отдельными полигонами
    polys.sort(key=lambda p: p.area, reverse=True)
    result = []
    for i, p in enumerate(polys):
        for q in polys[i + 1:]:
            if p.contains(q):
                p = p.difference(q)
        if p.area > 0:
            result.append(p)
    return result


def sections_along_axis(mesh: trimesh.Trimesh, axis: int, positions):
    """Все срезы вдоль одной оси за один вызов.

    mesh_multiplane режет меш сразу набором параллельных плоскостей и сам
    отдаёт 2D-отрезки. Это на порядок быстрее, чем звать mesh_plane в цикле,
    и полностью обходит Path3D, который падает на негерметичных мешах.
    """
    normal = np.zeros(3); normal[axis] = 1.0
    try:
        lines, _, _ = trimesh.intersections.mesh_multiplane(
            mesh, plane_origin=np.zeros(3), plane_normal=normal,
            heights=np.asarray(positions, dtype=float))
    except Exception:
        return []
    out = []
    for pos, seg in zip(positions, lines):
        polys = _polys_from_segments(seg)
        if polys:
            poly = max(polys, key=lambda q: q.area)
            if poly.area > 0:
                out.append((float(pos), poly, float(poly.area)))
    return out


def section_k(poly: Polygon):
    """K = r_вписанной / R_описанной для одного среза."""
    # упрощаем контур: на точность отношения радиусов это не влияет,
    # а точек в срезе сферы могут быть сотни
    p = poly.simplify(0.05, preserve_topology=True)
    if p.is_empty or p.area <= 0:
        p = poly

    if _FAST_R:
        R = float(shapely.minimum_bounding_radius(p))
    else:
        _, _, R = min_enclosing_circle(np.asarray(p.exterior.coords))
    if R < MIN_R_MM:
        return None

    if _FAST_r:
        try:
            seg = shapely.maximum_inscribed_circle(p, tolerance=R * 1e-3)
            r = float(seg.length)
        except Exception:
            r = max_inscribed_radius(p, R)
    else:
        r = max_inscribed_radius(p, R)

    return r / R if R > 0 else None


# ----------------------------------------------------------------------------
# АНАЛИЗ МЕША
# ----------------------------------------------------------------------------
def _extents_for_axes(vertices, R):
    """Габариты облака точек в системе осей R (3x3, строки — оси)."""
    local = vertices @ R.T
    lo, hi = local.min(axis=0), local.max(axis=0)
    return hi - lo, (hi + lo) / 2.0


def oriented_box(mesh: trimesh.Trimesh):
    """Ориентированный ограничивающий ящик товара.

    Сначала пробуем штатный OBB из trimesh (он точный, но требует рабочего
    qhull, то есть установленного scipy). Если qhull недоступен или падает,
    trimesh бросает "Points must be coplanar" — тогда считаем сами:
    главные оси через ковариацию (PCA), затем уточняем поворотом вокруг
    каждой оси с шагом 1 градус и берём вариант с минимальным объёмом.
    Возвращает (transform 4x4, extents 3).
    """
    try:
        obb = mesh.bounding_box_oriented
        return np.asarray(obb.primitive.transform, dtype=float), \
               np.asarray(obb.primitive.extents, dtype=float)
    except Exception:
        pass

    V = np.asarray(mesh.vertices, dtype=float)

    # для ПОИСКА ориентации хватает крайних точек: берём экстремумы
    # по множеству направлений (дешёвая замена выпуклой оболочке).
    # Финальные габариты потом всё равно меряем по всем вершинам.
    if len(V) > 2000:
        rng = np.random.default_rng(0)
        dirs = rng.normal(size=(96, 3))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        proj = V @ dirs.T
        idx = np.unique(np.concatenate([proj.argmax(axis=0), proj.argmin(axis=0)]))
        S = V[idx]
    else:
        S = V

    # главные оси
    cov = np.cov((V - V.mean(axis=0)).T)
    _, vecs = np.linalg.eigh(cov)
    R = vecs.T[::-1].copy()                       # строки = оси, по убыванию
    if np.linalg.det(R) < 0:
        R[2] *= -1.0

    best_R = R
    best_ext, _ = _extents_for_axes(S, R)
    best_vol = float(np.prod(best_ext))

    def rot(axis, deg):
        a = math.radians(deg)
        c, s_ = math.cos(a), math.sin(a)
        M = np.eye(3)
        i, j = [k for k in range(3) if k != axis]
        M[i, i] = c; M[i, j] = -s_
        M[j, i] = s_; M[j, j] = c
        return M

    # уточнение: два прохода по каждой оси — грубый и тонкий
    for step_deg, span in ((1.0, 90.0), (0.1, 1.0)):
        for axis in range(3):
            base = best_R.copy()
            base_vol = best_vol
            k = 0
            angles = np.arange(-span, span + 1e-9, step_deg) if step_deg < 1.0 \
                else np.arange(0.0, span, step_deg)
            for deg in angles:
                cand = rot(axis, float(deg)) @ base
                ext, _ = _extents_for_axes(S, cand)
                vol = float(np.prod(ext))
                if vol < best_vol:
                    best_vol, best_R = vol, cand

    # финальные габариты и центр — по всем вершинам, а не по подвыборке
    best_ext, best_ctr = _extents_for_axes(V, best_R)

    T = np.eye(4)
    T[:3, :3] = best_R.T                          # оси ящика -> мировые
    T[:3, 3] = best_R.T @ best_ctr                # центр ящика в мире
    return T, best_ext


def _pack(mesh, extents, best, per_axis_k, n_valid):
    dims = sorted([float(x) for x in extents], reverse=True)
    return {
        "dims_sorted": [round(d, 2) for d in dims],
        "k_max": round(best["k"], 4),
        "k_axis": best["axis"],
        "k_pos": best["pos"],
        "k_per_axis": per_axis_k,
        "sections_valid": n_valid,
        "volume": round(float(mesh.volume), 1) if mesh.is_volume else None,
        "watertight": bool(mesh.is_watertight),
    }


def size_fails(dims_sorted):
    """Габариты не проходят — форму можно не считать (приоритет правил)."""
    limits = sorted(MAX_BOX, reverse=True)
    return (min(dims_sorted) <= MIN_DIM
            or any(d > lim for d, lim in zip(dims_sorted, limits)))


def analyse_mesh(mesh: trimesh.Trimesh, step: float, early_exit: bool = True):
    """Возвращает словарь с габаритами и максимальным K по всем сечениям."""
    # 1. Собственные габариты: переводим меш в систему координат ящика
    transform, extents = oriented_box(mesh)
    extents = np.asarray(extents, dtype=float)

    best = {"k": 0.0, "axis": None, "pos": None, "area": None}
    per_axis_k = {}
    n_valid = 0

    # 2. Габариты проверяются первыми. Если товар не проходит по ним, он всё
    # равно уедет в негабарит — резать сечения бессмысленно, это чистая экономия.
    dims_now = sorted([float(x) for x in extents], reverse=True)
    if early_exit and size_fails(dims_now):
        return _pack(mesh, extents, best, {0: 0.0, 1: 0.0, 2: 0.0}, 0)

    m = mesh.copy()
    m.apply_transform(np.linalg.inv(transform))

    for axis in range(3):
        half = extents[axis] / 2.0
        # шаг: не мельче step, но и не больше ~60 сечений на ось
        n = int(max(3, min(60, math.ceil(extents[axis] / step))))
        # отступаем от торцов, чтобы не ловить вырожденные крайние срезы
        positions = np.linspace(-half * 0.98, half * 0.98, n)

        slices = sections_along_axis(m, axis, positions)
        if not slices:
            per_axis_k[axis] = 0.0
            continue

        max_area = max(s[2] for s in slices)
        axis_best = 0.0
        for pos, poly, area in slices:
            if area < MIN_AREA_FRAC * max_area:      # вырожденный срез — мимо
                continue
            k = section_k(poly)
            if k is None:
                continue
            n_valid += 1
            if k > axis_best:
                axis_best = k
            if k > best["k"]:
                best = {"k": float(k), "axis": int(axis),
                        "pos": round(pos, 2), "area": round(area, 1)}
            # критерий — "существует хотя бы одно сечение", поэтому как только
            # порог уверенно пробит, дальше искать нечего: категория не изменится
            if early_exit and k > K_THRESHOLD + K_TOL:
                per_axis_k[axis] = round(axis_best, 4)
                return _pack(mesh, extents, best, per_axis_k, n_valid)
        per_axis_k[axis] = round(axis_best, 4)

    return _pack(mesh, extents, best, per_axis_k, n_valid)


def classify(dims_sorted, k_max):
    """Приоритет по ТЗ: сначала габариты, потом форма."""
    L, W, H = dims_sorted
    limits = sorted(MAX_BOX, reverse=True)   # 450, 320, 320

    too_small = min(dims_sorted) <= MIN_DIM
    too_big = any(d > lim for d, lim in zip(dims_sorted, limits))

    # серые зоны
    near_min = abs(min(dims_sorted) - MIN_DIM) <= SIZE_TOL
    near_max = any(abs(d - lim) <= SIZE_TOL for d, lim in zip(dims_sorted, limits))
    borderline_size = bool(near_min or near_max)
    borderline_round = bool(abs(k_max - K_THRESHOLD) <= K_TOL)

    if too_small:
        return CAT_SIZE, "меньше минимально допустимых 10x10x10", borderline_size, borderline_round
    if too_big:
        return CAT_SIZE, "больше максимально допустимых 450x320x320", borderline_size, borderline_round
    if borderline_size:
        # консервативно: сомнение по габаритам -> из потока
        return CAT_SIZE, "габарит в серой зоне порога, решение консервативное", True, borderline_round

    if k_max > K_THRESHOLD:
        return CAT_ROUND, f"круг в сечении: K={k_max:.3f} > {K_THRESHOLD}", borderline_size, borderline_round
    if borderline_round:
        # консервативно: сомнение по форме -> доупаковка дешевле затора
        return CAT_ROUND, f"K={k_max:.3f} в серой зоне порога, решение консервативное", borderline_size, True

    return CAT_OK, f"габариты в норме, K={k_max:.3f}", borderline_size, borderline_round


# ----------------------------------------------------------------------------
# ГЛАВНОЕ
# ----------------------------------------------------------------------------
def load_mesh(path: Path, scale: float):
    obj = trimesh.load(str(path), force="mesh")
    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(tuple(obj.geometry.values()))
    if not isinstance(obj, trimesh.Trimesh) or obj.faces.shape[0] == 0:
        raise ValueError("пустой или неподдерживаемый меш")
    if scale != 1.0:
        obj.apply_scale(scale)
    obj.remove_infinite_values()
    obj.merge_vertices()
    obj.update_faces(obj.nondegenerate_faces())
    obj.update_faces(obj.unique_faces())
    obj.remove_unreferenced_vertices()
    if not obj.is_watertight:
        try:
            trimesh.repair.fill_holes(obj)
            trimesh.repair.fix_normals(obj)
        except Exception:
            pass
    return obj


def process_one(args_tuple):
    """Обработка одного файла. Вынесена наружу, чтобы работала в пуле процессов."""
    path, step, scale, early = args_tuple
    f = Path(path)
    t0 = time.time()
    try:
        mesh = load_mesh(f, scale)
        g = analyse_mesh(mesh, step, early_exit=early)
        cat, reason, b_size, b_round = classify(g["dims_sorted"], g["k_max"])
        row = {
            "file": f.name,
            "category": cat,
            "zone": {CAT_OK: "B", CAT_SIZE: "C", CAT_ROUND: "D"}[cat],
            "reason": reason,
            "dim_l": g["dims_sorted"][0],
            "dim_w": g["dims_sorted"][1],
            "dim_h": g["dims_sorted"][2],
            "k_max": g["k_max"],
            "k_axis": g["k_axis"],
            "k_pos": g["k_pos"],
            "borderline_size": int(b_size),
            "borderline_round": int(b_round),
            "volume_mm3": g["volume"],
            "watertight": int(g["watertight"]),
            "sections": g["sections_valid"],
            "seconds": round(time.time() - t0, 2),
        }
        detail = {**row, "path": str(f), "k_per_axis": g["k_per_axis"]}
        return row, detail
    except Exception as e:
        row = {"file": f.name, "category": "error", "zone": "R", "reason": str(e)}
        return row, {**row, "path": str(f)}


def main():
    ap = argparse.ArgumentParser(description="Разметка STL по правилам трека 3")
    ap.add_argument("--input", "-i", required=True, help="папка с STL")
    ap.add_argument("--output", "-o", default="./out", help="папка для результатов")
    ap.add_argument("--step", type=float, default=3.0,
                    help="шаг сечений в мм (мельче = точнее и медленнее)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="множитель единиц: 1.0 если STL в мм, 1000 если в метрах")
    ap.add_argument("--recursive", action="store_true", help="искать во вложенных папках")
    ap.add_argument("--no-early-exit", action="store_true",
                    help="считать все сечения даже когда категория уже ясна (медленно)")
    ap.add_argument("--jobs", "-j", type=int, default=0,
                    help="процессов параллельно (0 = по числу ядер)")
    args = ap.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*" if args.recursive else "*"
    files = sorted(p for p in in_dir.glob(pattern)
                   if p.is_file() and p.suffix.lower() in MESH_EXT)
    if not files:
        print(f"В {in_dir} не найдено мешей ({', '.join(sorted(MESH_EXT))})")
        sys.exit(1)

    print(f"Найдено файлов: {len(files)}   шаг сечений: {args.step} мм\n")

    rows, details = [], []
    counts = {CAT_OK: 0, CAT_SIZE: 0, CAT_ROUND: 0, "error": 0}
    early = not args.no_early_exit
    tasks = [(str(f), args.step, args.scale, early) for f in files]

    jobs = args.jobs if args.jobs > 0 else (os.cpu_count() or 1)
    jobs = max(1, min(jobs, len(files)))

    t_start = time.time()
    if jobs == 1:
        results = map(process_one, tasks)
    else:
        pool = cf.ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(process_one, tasks)
        print(f"Процессов: {jobs}\n")

    for idx, (row, detail) in enumerate(results, 1):
        rows.append(row)
        details.append(detail)
        cat = row["category"]
        counts[cat] = counts.get(cat, 0) + 1
        if cat == "error":
            print(f"[{idx:>3}/{len(files)}] {row['file']:<32} -> ОШИБКА: {row['reason']}")
        else:
            flag = "  [серая зона]" if (row["borderline_size"] or row["borderline_round"]) else ""
            dims = [row["dim_l"], row["dim_w"], row["dim_h"]]
            print(f"[{idx:>3}/{len(files)}] {row['file']:<32} -> {row['zone']} "
                  f"{cat:<13} {dims} K={row['k_max']:.3f}{flag}")
    elapsed = time.time() - t_start

    # ---- сохранение -------------------------------------------------------
    csv_path = out_dir / "results.csv"
    fields = ["file", "category", "zone", "reason", "dim_l", "dim_w", "dim_h",
              "k_max", "k_axis", "k_pos", "borderline_size", "borderline_round",
              "volume_mm3", "watertight", "sections", "seconds"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    json_path = out_dir / "results.json"
    payload = {
        "params": {
            "min_dim_mm": MIN_DIM, "max_box_mm": list(MAX_BOX),
            "k_threshold": K_THRESHOLD, "size_tol_mm": SIZE_TOL, "k_tol": K_TOL,
            "min_area_frac": MIN_AREA_FRAC, "min_r_mm": MIN_R_MM,
            "section_step_mm": args.step, "scale": args.scale,
        },
        "summary": counts,
        "items": details,
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    total = len(files)
    print("\n" + "-" * 62)
    print(f"B  подходит для сортировки : {counts[CAT_OK]:>4}  "
          f"({100*counts[CAT_OK]/total:.1f}%)")
    print(f"C  не подходит по габаритам: {counts[CAT_SIZE]:>4}  "
          f"({100*counts[CAT_SIZE]/total:.1f}%)")
    print(f"D  требует доупаковки      : {counts[CAT_ROUND]:>4}  "
          f"({100*counts[CAT_ROUND]/total:.1f}%)")
    print(f"R  ошибки чтения           : {counts['error']:>4}")
    grey = sum(1 for r in rows if r.get("borderline_size") or r.get("borderline_round"))
    print(f"   из них в серой зоне     : {grey:>4}")
    print("-" * 62)
    print(f"Время: {elapsed:.1f} с ({elapsed/total:.2f} с на объект)")
    print(f"Сохранено: {csv_path}\n           {json_path}")


if __name__ == "__main__":
    main()
