#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frames_to_cloud.py — сырые кадры трёх камер -> облако точек (визуальная оболочка).

Твоя часть контура: кадры -> силуэты -> оболочка -> облако, которое дальше
уходит в cloud_classify.classify_cloud().

Почему вычитание фона не такое простое, как кажется:
  - тень товара тоже отличается от фона и раздувает силуэт;
  - товар в цвет ленты даёт дыру в силуэте;
  - фон дрейфует (вибрация, пыль, свет) — один снимок устаревает;
  - жёсткий порог то ест края, то ловит шум.
Здесь всё это учтено: непрерывное обновление фона, адаптивный порог,
чистка морфологией, подавление тонких теней.

Две части:
  1. BackgroundModel — сегментация силуэта по кадру одной камеры.
  2. carve_cloud — пересечение трёх силуэтов в облако точек.

Калибровку камер (матрицы проекции) здесь считаем заданной: её делают один
раз по шахматной доске. Ниже — заглушка простой ортокамеры для проверки;
на железе подставляются реальные матрицы.
"""

import numpy as np
import cv2


# ======================================================================
# 1. СЕГМЕНТАЦИЯ СИЛУЭТА
# ======================================================================
class BackgroundModel:
    """Модель фона одной камеры с непрерывным обновлением.

    Держит медиану последних кадров пустой ленты. Силуэт = где текущий кадр
    заметно отличается от фона. Порог адаптивный (по разбросу самого фона),
    тени подавляются, мелкий шум убирается морфологией.
    """

    def __init__(self, backlit=True, learn=0.02):
        self.bg = None            # текущая оценка фона (float32)
        self.var = None           # разброс фона по пикселям
        self.backlit = backlit    # подсветка на просвет -> товар темнее фона
        self.learn = learn        # скорость обновления фона (0..1)

    def init_from(self, frames):
        """Инициализация по стопке кадров пустой ленты."""
        stack = np.stack([f.astype(np.float32) for f in frames], 0)
        self.bg = np.median(stack, 0)
        self.var = stack.std(0) + 1.0
        return self

    def update_background(self, frame, silhouette):
        """Обновляем фон ТАМ, ГДЕ НЕТ ТОВАРА — иначе товар «впечатается» в фон."""
        if self.bg is None:
            self.init_from([frame]); return
        m = (silhouette == 0)
        f = frame.astype(np.float32)
        self.bg[m] = (1 - self.learn) * self.bg[m] + self.learn * f[m]

    def silhouette(self, frame):
        """Силуэт товара в кадре: 1 = товар, 0 = фон."""
        if self.bg is None:
            raise RuntimeError("сначала init_from() по пустой ленте")
        f = frame.astype(np.float32)

        if f.ndim == 3:                       # цвет: разница по всем каналам
            diff = np.abs(f - self.bg).max(axis=2)
            bright_f = f.mean(axis=2)
            bright_bg = self.bg.mean(axis=2)
        else:
            diff = np.abs(f - self.bg)
            bright_f, bright_bg = f, self.bg

        var = self.var if self.var.ndim == 2 else self.var.mean(axis=2)
        mask = (diff > 3.0 * var).astype(np.uint8)   # адаптивный порог

        # Подавление тени. На просвет товар ТЕМНЕЕ фона сильно; тень — слабое
        # затемнение. Требуем, чтобы товар был заметно темнее (или на цветном
        # фоне — отличался по цвету, а не только по яркости).
        if self.backlit:
            deep = (bright_f < 0.6 * bright_bg).astype(np.uint8)
            mask &= deep

        # чистка: закрыть дырки в товаре, убрать крапинки на фоне
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # оставляем только крупнейшую связную область (товар — один объект)
        n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mask = (lbl == biggest).astype(np.uint8)
        return mask


# ======================================================================
# 2. ВЫРЕЗАНИЕ ОБЛАКА ИЗ ТРЁХ СИЛУЭТОВ
# ======================================================================
def carve_cloud(silhouettes, cameras, bounds, pitch=2.0):
    """Пересечение силуэтов трёх камер -> облако точек поверхности оболочки.

    silhouettes: список масок (H, W) uint8, по одной на камеру.
    cameras:     список функций project(P)->(u,v,visible) для каждой камеры.
                 На железе это реальные матрицы проекции из калибровки.
    bounds:      (xmin, xmax, ymin, ymax, zmin, zmax) рабочего объёма, мм.
    pitch:       размер вокселя, мм.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    xs = np.arange(xmin, xmax, pitch)
    ys = np.arange(ymin, ymax, pitch)
    zs = np.arange(zmin, zmax, pitch)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    alive = np.ones(len(P), dtype=bool)
    for sil, project in zip(silhouettes, cameras):
        u, v, vis = project(P)
        H, W = sil.shape
        ui = np.round(u).astype(np.int32)
        vi = np.round(v).astype(np.int32)
        inside = np.zeros(len(P), dtype=bool)
        ok = vis & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        inside[ok] = sil[vi[ok], ui[ok]] > 0
        alive &= inside
        if not alive.any():
            break

    occ = alive.reshape(X.shape)
    if not occ.any():
        return np.empty((0, 3))

    # поверхность оболочки: воксели, у которых есть пустой сосед.
    # внутренние точки классификатору не нужны, только оболочка.
    from scipy.ndimage import binary_erosion
    surface = occ & ~binary_erosion(occ)
    idx = np.argwhere(surface)
    cloud = np.stack([xs[idx[:, 0]], ys[idx[:, 1]], zs[idx[:, 2]]], axis=1)
    return cloud


# ======================================================================
# ЗАГЛУШКА КАМЕРЫ ДЛЯ ПРОВЕРКИ БЕЗ ЖЕЛЕЗА (орто-проекция)
# ======================================================================
def ortho_camera(axis, img_shape, bounds, flip=False):
    """Простейшая орто-камера вдоль оси. На железе заменяется калиброванной."""
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    H, W = img_shape
    a = {"x": 0, "y": 1, "z": 2}[axis]
    keep = [i for i in range(3) if i != a]
    lo = np.array([bounds[2 * keep[0]], bounds[2 * keep[1]]])
    hi = np.array([bounds[2 * keep[0] + 1], bounds[2 * keep[1] + 1]])

    def project(P):
        q = P[:, keep]
        uv = (q - lo) / (hi - lo)
        u = uv[:, 0] * (W - 1)
        v = (1 - uv[:, 1]) * (H - 1) if flip else uv[:, 1] * (H - 1)
        vis = np.ones(len(P), dtype=bool)
        return u, v, vis
    return project


if __name__ == "__main__":
    # Самопроверка на синтетике: рендерим цилиндр в три силуэта,
    # вырезаем оболочку, классифицируем.
    import importlib.util
    from pathlib import Path
    import trimesh, warnings
    warnings.filterwarnings("ignore")

    def load(name):
        s = importlib.util.spec_from_file_location(name, str(Path(__file__).parent / f"{name}.py"))
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
    cc = load("cloud_classify")

    bounds = (-150, 150, -150, 150, 0, 300)
    mesh = trimesh.creation.cylinder(radius=45, height=250, sections=64)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    mesh.apply_translation([0, 0, -mesh.bounds[0][2]])

    # рендерим "истинные" силуэты орто-камерами (сверху и две боковые)
    def render_sil(axis, flip=False, shape=(240, 240)):
        proj = ortho_camera(axis, shape, bounds, flip)
        pts = mesh.sample(60000)
        u, v, _ = proj(pts)
        H, W = shape
        img = np.zeros(shape, np.uint8)
        ui = np.clip(np.round(u).astype(int), 0, W-1)
        vi = np.clip(np.round(v).astype(int), 0, H-1)
        img[vi, ui] = 1
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))

    sils = [render_sil("z"), render_sil("y"), render_sil("x")]
    cams = [ortho_camera("z", (240,240), bounds),
            ortho_camera("y", (240,240), bounds),
            ortho_camera("x", (240,240), bounds)]
    cloud = carve_cloud(sils, cams, bounds, pitch=3.0)
    print("точек в облаке:", len(cloud))
    r = cc.classify_cloud(cloud, source="hull")
    print("зона:", r["zone"], "| K:", r["k_max"], "| габариты:", r["dims_mm"])
    print("ожидалось: зона D (цилиндр лёжа)")
