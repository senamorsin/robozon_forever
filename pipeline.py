#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — ЕДИНАЯ ТОЧКА ВХОДА для интеграции на Raspberry Pi.

Сокомандник вызывает отсюда ровно две вещи:
  1. Vision(config)         — создать один раз при старте (загружает фон, калибровку).
  2. vision.process(...)    — на каждый товар: кадры + датчики -> решение.

Всё остальное (силуэты, оболочка, критерий круга, поза захвата, укатывание)
происходит внутри. Наружу выходит один словарь Decision — бери и командуй
контроллером.

Границы ответственности (кто за что):
  ТВОИ скрипты отвечают за:   класс товара, габариты, K, поза захвата, флаг
                              укатывания, флаг "не уверен -> разбор".
  КОНТРОЛЛЕР отвечает за:      тайминг отвода по счётчику ленты, физику приводов,
                              что делать с зоной (B/C/D/R), аварийные сбросы.

То есть process() говорит ЧТО за товар и КУДА его, а КОГДА и КАК — на контроллере.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

# твои модули
import frames_to_cloud as f2c
import cloud_classify as cc


# ======================================================================
# КОНФИГ — заполняется один раз при старте
# ======================================================================
@dataclass
class VisionConfig:
    # рабочий объём измерения, мм: (xmin,xmax, ymin,ymax, zmin,zmax)
    bounds: tuple = (-260, 260, -260, 260, 0, 520)
    voxel_mm: float = 3.0             # размер вокселя оболочки (меньше=точнее=медленнее)
    source: str = "hull"             # "hull"=только камеры, "surface"=есть лазер
    backlit: bool = True             # камеры с подсветкой на просвет
    size_tol_mm: float = 5.0         # допуск габаритов (уточнён у организаторов)
    belt_speed_mm_s: float = 1000.0  # скорость ленты, задана условиями
    # матрицы проекции камер: список функций project(P)->(u,v,visible).
    # На железе кладутся сюда после калибровки по шахматной доске.
    cameras: list = field(default_factory=list)


# ======================================================================
# РЕЗУЛЬТАТ — то, что получает контроллер
# ======================================================================
@dataclass
class Decision:
    ok: bool                 # удалось ли принять решение (False -> разбор)
    zone: str                # "B" | "C" | "D" | "R"  — куда物 товар
    category: str            # sortable | no_fit_size | needs_repack | error
    reason: str              # человекочитаемая причина (для лога)
    dims_mm: list = field(default_factory=list)   # [L, W, H]
    k_max: float = 0.0       # показатель круглости
    confidence: float = 1.0  # 0..1, ниже порога -> контроллер шлёт в разбор
    borderline: bool = False # у порога, решено консервативно
    grasp: Optional[dict] = None   # поза для клешни, только для зоны D
    rolling: bool = False    # товар катится (краевой случай)
    debug: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


# ======================================================================
# ГЛАВНЫЙ КЛАСС
# ======================================================================
class Vision:
    """Один экземпляр на всё время работы. Держит модели фона трёх камер."""

    def __init__(self, config: VisionConfig, empty_belt_frames=None):
        self.cfg = config
        self.bg_models = None
        if empty_belt_frames is not None:
            self.calibrate_background(empty_belt_frames)

    def calibrate_background(self, empty_belt_frames):
        """empty_belt_frames: список из 3 стопок кадров пустой ленты (по камере)."""
        self.bg_models = [
            f2c.BackgroundModel(backlit=self.cfg.backlit).init_from(stack)
            for stack in empty_belt_frames
        ]

    # ------------------------------------------------------------------
    def process(self, frames=None, cloud=None, track=None) -> Decision:
        """Главный вызов на один товар.

        Варианты входа (используй тот, что доступен на железе):
          frames : список из 3 кадров с камер (BGR или grayscale numpy).
                   Тогда силуэты и облако строятся здесь.
          cloud  : готовое облако точек (N,3) в мм — если ты строишь его
                   где-то ещё (например, прямо с лазера). frames тогда не нужен.
          track  : опционально, список (t_сек, x_центра_мм) для детектора
                   укатывания. Если None — флаг rolling всегда False.

        Возвращает Decision. Контроллер смотрит на .zone и .grasp.
        """
        # 1. получить облако точек
        if cloud is None:
            if frames is None:
                return Decision(False, "R", "error", "нет ни кадров, ни облака",
                                confidence=0.0)
            if self.bg_models is None:
                return Decision(False, "R", "error",
                                "фон не откалиброван (calibrate_background)",
                                confidence=0.0)
            cloud = self._frames_to_cloud(frames)

        cloud = np.asarray(cloud, float)
        if len(cloud) < 30:
            return Decision(False, "R", "error", "слишком мало точек",
                            confidence=0.0, debug={"npoints": int(len(cloud))})

        # 2. классификация (твоя основная логика)
        r = cc.classify_cloud(cloud, source=self.cfg.source, track=track,
                              size_tol=self.cfg.size_tol_mm)

        # 3. уверенность: у порога или мало точек -> ниже единицы
        conf = 1.0
        if r.get("borderline_size") or r.get("borderline_round"):
            conf = 0.6
        if len(cloud) < 200:
            conf = min(conf, 0.5)

        roll = bool(r.get("roll", {}).get("rolling", False))

        return Decision(
            ok=(r["category"] != "error"),
            zone=r["zone"],
            category=r["category"],
            reason=r["reason"],
            dims_mm=r.get("dims_mm", []),
            k_max=r.get("k_max", 0.0),
            confidence=conf,
            borderline=bool(r.get("borderline_size") or r.get("borderline_round")),
            grasp=r.get("grasp"),
            rolling=roll,
            debug={"npoints": int(len(cloud)),
                   "k_vertical": r.get("k_vertical"),
                   "k_horizontal": r.get("k_horizontal"),
                   "source": r.get("source")},
        )

    # ------------------------------------------------------------------
    def _frames_to_cloud(self, frames):
        sils = [m.silhouette(f) for m, f in zip(self.bg_models, frames)]
        cloud = f2c.carve_cloud(sils, self.cfg.cameras, self.cfg.bounds,
                                pitch=self.cfg.voxel_mm)
        # обновляем фон там, где нет товара (чтобы не устаревал)
        for m, f, s in zip(self.bg_models, frames, sils):
            m.update_background(f, s)
        return cloud


# ======================================================================
# САМОПРОВЕРКА без железа
# ======================================================================
def _lying_cyl():
    import trimesh, numpy as np
    m = trimesh.creation.cylinder(radius=45, height=250, sections=64)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    return m


if __name__ == "__main__":
    import trimesh, warnings
    warnings.filterwarnings("ignore")

    bounds = (-150, 150, -150, 150, 0, 300)
    cfg = VisionConfig(bounds=bounds, voxel_mm=3.0, source="hull",
                       cameras=[f2c.ortho_camera("z", (240, 240), bounds),
                                f2c.ortho_camera("y", (240, 240), bounds),
                                f2c.ortho_camera("x", (240, 240), bounds)])
    vis = Vision(cfg)

    # эмулируем товар: строим облако напрямую (как будто уже с сенсора)
    import importlib.util
    from pathlib import Path
    s = importlib.util.spec_from_file_location("vs", str(Path(__file__).parent / "virtual_scan.py"))
    vs = importlib.util.module_from_spec(s); s.loader.exec_module(vs)

    for name, mk in [
        ("цилиндр лёжа", lambda: _lying_cyl()),
        ("коробка", lambda: trimesh.creation.box(extents=[300, 200, 150])),
        ("негабарит", lambda: trimesh.creation.box(extents=[480, 400, 300])),
    ]:
        m = mk()
        m.apply_translation([0, 0, -m.bounds[0][2]])
        cloud = vs.scan_from_above(m, res=0.5)
        # для укатывания подсунем трек: коробка едет с лентой, цилиндр катится
        track = None
        if "цилиндр" in name:
            track = [(0.0, 100), (0.1, 260), (0.2, 440)]   # быстрее ленты
        d = vis.process(cloud=cloud, track=track)
        print(f"{name:<14} -> зона {d.zone}  {d.category:<12} "
              f"conf={d.confidence}  roll={d.rolling}  dims={d.dims_mm}")

