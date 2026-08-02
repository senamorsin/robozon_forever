#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
link_protocol.py — протокол связи Raspberry Pi <-> ESP32 (сторона Pi).

Граница ответственности:
  ESP32  отвечает за ВРЕМЯ: ловит товар по фотобарьеру и энкодеру, присваивает
         ID, командует стробом, отводами, клешнёй, интерлоки.
  Pi     отвечает за СМЫСЛ: получает кадры, строит облако, классифицирует,
         возвращает ESP решение по каждому ID.

Через UART идут короткие строчные сообщения (не байтовая упаковка) — их легко
читать в терминале при отладке и одинаково просто парсить на обеих сторонах.

=====================================================================
 ФОРМАТ СООБЩЕНИЯ  (одна строка, поля через ';', конец '\\n')
=====================================================================
   TYPE;key=value;key=value;...;CRC=XX\\n

 CRC — младший байт XOR всех символов строки до "CRC=" включительно
 (в hex, 2 знака). Ловит битые байты на линии. Можно временно слать
 CRC=00 и не проверять, пока отлаживаете.

---------------------------------------------------------------------
 ESP32  ->  Pi
---------------------------------------------------------------------
 SHOOT;id=<int>;enc=<int>                # товар в точке съёмки, снимай
     id  — номер товара (ESP присвоил при пересечении барьера)
     enc — показание энкодера в момент вспышки (мм), для трекинга

 TRACK;id=<int>;enc=<int>                # опц.: позиция товара по кадрам
     шлётся несколько раз, если нужен детектор укатывания

 STATUS;state=<ready|busy|estop>        # состояние линии
 PONG;t=<int>                            # ответ на PING

---------------------------------------------------------------------
 Pi  ->  ESP32
---------------------------------------------------------------------
 RESULT;id=<int>;zone=<B|C|D|R>;conf=<0..100>;roll=<0|1>[;углы серв]
     zone  — куда физически отправить товар
     conf  — уверенность в процентах; ESP сам решает порог -> R
     roll  — товар катится (ESP может сбросить в R или на повтор)
     для zone=D добавляются ГОТОВЫЕ углы серв (Pi посчитал кинематику):
       s0=<int>;s1=<int>;s2=<int>;grip=<int>   — база, плечо, локоть, клешня
     Кинематику Pi считает у себя — Arduino остаётся тупым и быстрым, и через
     одну линию не смешиваются два несовместимых формата.

 PING;t=<int>                            # проверка живости линии
 ACK;id=<int>                            # подтверждение приёма SHOOT

=====================================================================
 ПРИМЕРЫ ОБМЕНА
=====================================================================
 ESP -> Pi:  SHOOT;id=1042;enc=385210;CRC=3A
 Pi  -> ESP: ACK;id=1042;CRC=7F
 Pi  -> ESP: RESULT;id=1042;zone=D;conf=95;roll=0;gx=12.3;gy=-4.1;gaxis=37;gopen=100;gz=90;CRC=5C
 ESP -> Pi:  SHOOT;id=1043;enc=386020;CRC=2B
 Pi  -> ESP: RESULT;id=1043;zone=B;conf=100;roll=0;CRC=44

Порт по умолчанию: /dev/ttyAMA0 (UART Pi 5), 115200 бод, 8N1.
"""

import time

try:
    import serial            # pip install pyserial (на Pi)
except ImportError:
    serial = None            # позволяем импортировать модуль и без железа


# ---------------------------------------------------------------- CRC
def _crc(payload: str) -> str:
    """Младший байт XOR всех символов, 2 знака hex."""
    x = 0
    for ch in payload:
        x ^= ord(ch)
    return f"{x & 0xFF:02X}"


def encode(msg_type: str, **fields) -> str:
    """Собрать строку сообщения с CRC и переводом строки."""
    body = msg_type
    for k, v in fields.items():
        body += f";{k}={v}"
    body += ";CRC="
    return body + _crc(body) + "\n"


def decode(line: str):
    """Разобрать строку в (тип, словарь полей). CRC проверяется.

    Возвращает (type, fields) или (None, {"error": ...}) при сбое.
    """
    line = line.strip()
    if not line:
        return None, {"error": "пустая строка"}
    if ";CRC=" not in line:
        return None, {"error": "нет CRC"}
    body, got = line.rsplit("CRC=", 1)
    body += "CRC="
    if _crc(body) != got.strip():
        return None, {"error": f"CRC не сошлась: ждали {_crc(body)}, пришло {got}"}

    parts = body[:-4].rstrip(";").split(";")   # убрать хвост ';CRC='
    msg_type = parts[0]
    fields = {}
    for p in parts[1:]:
        if "=" in p:
            k, v = p.split("=", 1)
            fields[k] = v
    return msg_type, fields


# ---------------------------------------------------------------- сторона Pi
class Link:
    """UART-линия на стороне Raspberry Pi.

    Использование в рантайме:
        link = Link("/dev/ttyAMA0")
        for msg_type, f in link.listen():
            if msg_type == "SHOOT":
                id_ = int(f["id"])
                decision = vision.process(frames=grab_frames())  # твой pipeline
                link.send_result(id_, decision)
    """

    def __init__(self, port="/dev/ttyAMA0", baud=115200, timeout=0.05):
        if serial is None:
            raise RuntimeError("нет pyserial: pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=timeout)
        self._buf = ""

    # --- приём ---
    def listen(self):
        """Генератор входящих сообщений (type, fields). Не блокирует надолго."""
        while True:
            chunk = self.ser.read(256).decode("ascii", errors="ignore")
            if chunk:
                self._buf += chunk
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    t, f = decode(line)
                    if t is not None:
                        yield t, f
                    # битые строки молча пропускаем — ESP пришлёт заново
            else:
                yield None, None          # тайм-аут, можно заняться другим

    # --- передача ---
    def _write(self, s: str):
        self.ser.write(s.encode("ascii"))

    def send_ack(self, id_: int):
        self._write(encode("ACK", id=id_))

    def send_result(self, id_: int, decision, servo=None):
        """decision — объект Decision из pipeline.py (или совместимый dict).
        servo — кортеж (s0,s1,s2,grip) готовых углов серв для зоны D
                (конвейер считает их кинематикой перед вызовом)."""
        d = decision.as_dict() if hasattr(decision, "as_dict") else decision
        fields = {
            "id": id_,
            "zone": d.get("zone", "R"),
            "conf": int(round(d.get("confidence", 0.0) * 100)),
            "roll": 1 if d.get("rolling") else 0,
        }
        if d.get("zone") == "D" and servo is not None:
            s0, s1, s2, grip = servo
            fields.update({"s0": int(round(s0)), "s1": int(round(s1)),
                           "s2": int(round(s2)), "grip": int(round(grip))})
        self._write(encode("RESULT", **fields))

    def send_ping(self):
        self._write(encode("PING", t=int(time.time() * 1000)))


def _axis_deg(axis_xy):
    """Вектор оси -> угол в градусах 0..180 (клешне направление, не знак)."""
    import math
    a = math.degrees(math.atan2(axis_xy[1], axis_xy[0])) % 180.0
    return a


# ---------------------------------------------------------------- самопроверка
if __name__ == "__main__":
    # round-trip без железа: закодировали -> раскодировали -> сверили
    line = encode("SHOOT", id=1042, enc=385210)
    print("строка:", line.strip())
    t, f = decode(line)
    print("разбор:", t, f)
    assert t == "SHOOT" and f["id"] == "1042" and f["enc"] == "385210"

    # RESULT для круглого товара с позой
    fake = {"zone": "D", "confidence": 0.95, "rolling": False,
            "grasp": {"center_xy": [12.3, -4.1], "grip_axis": [0.8, 0.6],
                      "open_mm": 100.0, "top_z": 90.0}}
    class D:
        def as_dict(self): return fake
    import io
    # проверим сборку RESULT-строки напрямую
    from math import atan2, degrees
    res = encode("RESULT", id=1042, zone="D", conf=95, roll=0,
                 gx=12.3, gy=-4.1, gaxis=37, gopen=100, gz=90)
    print("RESULT:", res.strip())
    t2, f2 = decode(res)
    assert t2 == "RESULT" and f2["zone"] == "D" and f2["gopen"] == "100"

    # порча байта -> CRC ловит
    bad = res.replace("zone=D", "zone=B")
    t3, f3 = decode(bad)
    print("битая строка поймана:", t3 is None, "|", f3.get("error"))
    assert t3 is None

    print("\nвсе проверки прошли")
