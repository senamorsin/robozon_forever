#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
runtime.py — МНОГОПОТОЧНЫЙ КОНВЕЙЕР-РАНТАЙМ на Raspberry Pi.

Собран из многопоточного скелета техника + наше зрение + защита "рука занята".

Три потока (как у техника):
  reader  — читает Serial от Arduino, кладёт строки в очередь;
  worker  — тяжёлое зрение и кинематика, шлёт команды руке;
  grabber — непрерывный захват кадров с камер.
Разведение чтения и вычисления не даёт потерять SHOOT, пока Pi считает облако.

Защита "одна рука — один товар" (через квитанцию DONE):
  Рука физически кладёт товар несколько секунд. Пока она занята, слать ей
  вторую SEQ нельзя. Поэтому:
    - Arduino по завершении шлёт "DONE;id=N" -> рука свободна;
    - Pi держит флаг arm_busy; пока он стоит, новые товары НЕ отправляются
      руке, а честно уходят в разбор (рука всё равно не успела бы их взять).
  Так очередь не копится и рука не получает команду посреди движения.

Обмен:
  Arduino -> Pi:  SHOOT;id=N   DONE;id=N
  Pi -> Arduino:  SEQ;id=N;zone=C;pk=..;lf=..;dp=..

Запуск на железе:   python3 runtime.py --port /dev/ttyACM0
Проверка без железа: python3 runtime.py --selftest
"""

import argparse
import time
import threading
import queue

import numpy as np

import pipeline
import arm_kinematics as ak


# ############################################################################
# #   ПАРАМЕТРЫ УПРАВЛЕНИЯ В ПОТОКЕ — МЕНЯТЬ ЗДЕСЬ                            #
# #   (геометрия руки и точки сброса — в arm_kinematics.py)                  #
# ############################################################################

PICK_OFFSET_X = 0.0       # мм: перенос точки захвата из координат зрения
PICK_OFFSET_Y = 180.0     #     в координаты руки (где стоит стол)
PICK_OFFSET_Z = 20.0

USE_VISION_PICK_POINT = False   # True: рука тянется туда, где реально лёг товар
CONFIDENCE_REJECT = 0.5         # ниже этой уверенности -> зона R

# Страховка на случай, если DONE потерялся: если рука "занята" дольше этого
# времени, считаем её освободившейся принудительно (чтобы не зависнуть навсегда).
ARM_BUSY_TIMEOUT_S = 12.0

MAX_QUEUE = 50            # предел очереди команд, чтобы не росла бесконечно

# ############################################################################


class AsyncFrameGrabber(threading.Thread):
    """Поток непрерывного захвата кадров с камер (код техника)."""
    def __init__(self, camera_source):
        super().__init__(daemon=True)
        self.camera_source = camera_source
        self.latest_frames = None
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            if self.camera_source is not None:
                frames = self.camera_source.read()
                with self.lock:
                    self.latest_frames = frames
            time.sleep(0.005)

    def get_frames(self):
        with self.lock:
            return self.latest_frames


class ThreadedRuntime:
    def __init__(self, vision: pipeline.Vision, frame_grabber=None,
                 conf_reject=CONFIDENCE_REJECT):
        self.vision = vision
        self.frame_grabber = frame_grabber
        self.conf_reject = conf_reject

        self.tracks = {}
        self.tracks_lock = threading.Lock()
        self.cmd_queue = queue.Queue(maxsize=MAX_QUEUE)
        self.serial_lock = threading.Lock()

        # --- состояние руки ---
        self.arm_busy = False
        self.arm_busy_since = 0.0
        self.arm_lock = threading.Lock()

    # ------------------------------------------------------------------
    def _arm_is_free(self):
        """Свободна ли рука. С защитой от потерянного DONE по таймауту."""
        with self.arm_lock:
            if self.arm_busy and (time.time() - self.arm_busy_since) > ARM_BUSY_TIMEOUT_S:
                self.arm_busy = False        # DONE потерялся — освобождаем сами
            return not self.arm_busy

    def _mark_arm_busy(self):
        with self.arm_lock:
            self.arm_busy = True
            self.arm_busy_since = time.time()

    def _mark_arm_free(self):
        with self.arm_lock:
            self.arm_busy = False

    # ------------------------------------------------------------------
    def decide(self, id_, cloud=None):
        """Зрение: категория + траектория руки (или None для B)."""
        with self.tracks_lock:
            track = self.tracks.get(id_)

        if cloud is not None:
            decision = self.vision.process(cloud=cloud, track=track)
        else:
            frames = self.frame_grabber.get_frames() if self.frame_grabber else None
            decision = self.vision.process(frames=frames, track=track)

        zone = decision.zone
        if decision.confidence < self.conf_reject and zone != "B":
            zone = "R"

        if zone == "B":
            return "B", None, decision

        pick_point = None
        if USE_VISION_PICK_POINT and decision.grasp is not None:
            gx, gy = decision.grasp["center_xy"]
            pick_point = {
                "x": float(gx) + PICK_OFFSET_X,
                "y": float(gy) + PICK_OFFSET_Y,
                "z": float(decision.grasp["top_z"]) + PICK_OFFSET_Z,
            }
        try:
            trajectory = ak.calculate_trajectory(zone, pick_point=pick_point)
        except ValueError:
            try:
                trajectory = ak.calculate_trajectory("R", pick_point=pick_point)
                zone = "R"
                decision.reason = "рука не дотянулась, товар в разбор"
            except ValueError:
                return "R", None, decision
        return zone, trajectory, decision

    @staticmethod
    def format_seq(id_, zone, trajectory):
        pk, lf, dp = trajectory
        return (f"SEQ;id={id_};zone={zone};"
                f"pk={pk[0]},{pk[1]},{pk[2]};"
                f"lf={lf[0]},{lf[1]},{lf[2]};"
                f"dp={dp[0]},{dp[1]},{dp[2]}\n")

    # ------------------------------------------------------------------
    def _serial_reader_thread(self, ser):
        buf = ""
        while True:
            try:
                if ser.in_waiting > 0:
                    buf += ser.read(ser.in_waiting).decode("utf-8", errors="ignore")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line:
                            try:
                                self.cmd_queue.put_nowait(line)
                            except queue.Full:
                                pass          # очередь забита — дропаем, не копим
                else:
                    time.sleep(0.001)
            except Exception:
                time.sleep(0.01)

    def _worker_thread(self, ser):
        while True:
            try:
                line = self.cmd_queue.get(timeout=0.1)
                self._handle_line(line, ser)
                self.cmd_queue.task_done()
            except queue.Empty:
                continue

    def _handle_line(self, line, ser):
        # рука закончила -> свободна
        if line.startswith("DONE"):
            self._mark_arm_free()
            print(f"[Arm] DONE id={_parse_id(line)} — рука свободна")
            return

        if line.startswith("SHOOT"):
            id_ = _parse_id(line)
            zone, trajectory, decision = self.decide(id_)

            # B: рука не нужна, товар едет в сортировщик
            if zone == "B":
                print(f"[Worker] id={id_} -> B (сортировщик)")
                self._forget(id_)
                return

            # C/D/R: нужна рука. Занята -> товар не взять, он проезжает в разбор
            if not self._arm_is_free():
                print(f"[Worker] id={id_} -> {zone}, но РУКА ЗАНЯТА — пропуск в разбор")
                self._forget(id_)
                return

            if trajectory is not None:
                self._mark_arm_busy()
                cmd = self.format_seq(id_, zone, trajectory)
                with self.serial_lock:
                    ser.write(cmd.encode("utf-8"))
                print(f"[Worker] id={id_} -> {zone}  {cmd.strip()}")
            self._forget(id_)

        elif line.startswith("TRACK"):
            id_ = _parse_id(line)
            x = _parse_field(line, "x", 0.0)
            with self.tracks_lock:
                self.tracks.setdefault(id_, []).append((time.time(), x))

    def _forget(self, id_):
        with self.tracks_lock:
            self.tracks.pop(id_, None)

    def run(self, ser):
        reader = threading.Thread(target=self._serial_reader_thread, args=(ser,), daemon=True)
        worker = threading.Thread(target=self._worker_thread, args=(ser,), daemon=True)
        if self.frame_grabber:
            self.frame_grabber.start()
        reader.start()
        worker.start()
        print("[Runtime] Потоки чтения UART, захвата кадров и Vision запущены.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Runtime] Остановка конвейера.")


# ---------------------------------------------------------------- утилиты
def _parse_id(line):
    return int(_parse_field(line, "id", 0))

def _parse_field(line, key, default):
    for part in line.split(";"):
        if part.startswith(key + "="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return default
    return default


# ---------------------------------------------------------------- фабрика
def build_runtime(camera_source=None):
    import frames_to_cloud as f2c
    bounds = (-260, 260, -260, 260, 0, 520)
    cfg = pipeline.VisionConfig(
        bounds=bounds, voxel_mm=3.0, source="hull", size_tol_mm=5.0,
        cameras=[f2c.ortho_camera("z", (240, 240), bounds),
                 f2c.ortho_camera("y", (240, 240), bounds),
                 f2c.ortho_camera("x", (240, 240), bounds)],
    )
    vision = pipeline.Vision(cfg)
    grabber = AsyncFrameGrabber(camera_source) if camera_source else None
    return ThreadedRuntime(vision, frame_grabber=grabber)


# ---------------------------------------------------------------- самопроверка
def selftest():
    import trimesh, warnings, importlib.util
    from pathlib import Path
    warnings.filterwarnings("ignore")
    s = importlib.util.spec_from_file_location("vs", str(Path(__file__).parent / "virtual_scan.py"))
    vs = importlib.util.module_from_spec(s); s.loader.exec_module(vs)

    rt = build_runtime()

    def lying_cyl():
        m = trimesh.creation.cylinder(radius=45, height=250, sections=64)
        m.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        return m

    cases = [
        ("коробка",         trimesh.creation.box(extents=[300, 200, 150]), None),
        ("негабарит",       trimesh.creation.box(extents=[480, 400, 300]), None),
        ("цилиндр D",       lying_cyl(), None),
        ("цилиндр катится", lying_cyl(), [(0.0, 100), (0.1, 260), (0.2, 440)]),
    ]

    print(f"{'товар':<18}{'зона':>6}  команда на Arduino")
    print("-" * 72)
    for i, (name, mesh, track) in enumerate(cases, 1):
        mesh.apply_translation([0, 0, -mesh.bounds[0][2]])
        cloud = vs.scan_from_above(mesh, res=0.5)
        if track:
            rt.tracks[i] = list(track)
        zone, trajectory, decision = rt.decide(i, cloud=cloud)
        if zone == "B":
            cmd = "(рука не нужна, в сортировщик)"
        elif trajectory is not None:
            cmd = rt.format_seq(i, zone, trajectory).strip()
        else:
            cmd = "(пропуск)"
        print(f"{name:<18}{zone:>6}  {cmd}")

    # проверка защиты "рука занята"
    print("\nпроверка блокировки занятой руки:")
    rt._mark_arm_busy()
    print("  рука занята ->", "свободна" if rt._arm_is_free() else "ЗАНЯТА (новый товар пройдёт в разбор)")
    rt._mark_arm_free()
    print("  после DONE ->", "СВОБОДНА (берёт следующий)" if rt._arm_is_free() else "занята")

    print("\nконвейер прошёл: зрение -> категория -> траектория -> SEQ, с защитой руки")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
    else:
        import serial
        rt = build_runtime(camera_source=None)   # camera_source подставит владелец камер
        ser = serial.Serial(args.port, 115200, timeout=0.01)
        time.sleep(2)
        ser.reset_input_buffer()
        rt.run(ser)
