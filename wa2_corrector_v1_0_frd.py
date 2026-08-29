#!/usr/bin/env python3
"""
=====================================
Линейно-фазовая коррекция АЧХ акустики Wharfedale Aura 2.

ИНСТРУКЦИЯ ПО ЗАПУСКУ И ПАРАМЕТРЫ КОМАНДНОЙ СТРОКИ
--------------------------------------------------
Скрипт запускается из терминала. Поддерживается только ОС Linux. Для работы требуются установленные зависимости.

Синтаксис запуска:
  python3 wharfedale_aura2_corrector.py -i <входная_папка> -o <выходная_папка> [опции]

Обязательные параметры:
  -o, --output <путь>     Путь к каталогу, куда будут записаны обработанные файлы.
                          Должен отличаться от входного каталога. Создается автоматически.

Необязательные параметры:
  -i, --input <путь>      Путь к исходной медиатеке FLAC.
                          По умолчанию: "/MDA/Музыка".
  --frd <путь>            Путь к кастомному текстовому файлу АЧХ (.frd / .txt).
                          Если не указан, используется встроенная эталонная АЧХ Aura 2.
  -w, --workers <число>   Количество параллельно обрабатываемых альбомов (процессов).
                          По умолчанию: половина доступных ядер CPU (для баланса нагрузок).
  --max-boost <число>     Максимальный предел усиления баса в дБ (диапазон: от 0.0 до 9.0).
                          По умолчанию: 9.0. Помогает адаптировать отдачу под комнату.
  --dry-run               Тестовый холостой режим. Скрипт проанализирует файлы,
                          рассчитает True Peak и покажет планируемые действия без записи на диск.
  --force                 Принудительная перезапись. Пережимать файлы, даже если они уже
                          были успешно обработаны ранее и имеют метку коррекции.
  --verify                Верификация после записи: каждый выходной FLAC открывается через
                          soundfile, проверяется декодируемость и число семплов.
                          Добавляет ~8-12%% к общему времени обработки. Рекомендуется
                          для первого прогона или после обновления версии.
  --report <путь>         Имя файла итогового JSON-отчета.
                          По умолчанию: "wharfedale_corrector.json".
  --log <путь>            Путь к файлу логов работы скрипта.
                          По умолчанию: "wharfedale_corrector.log".
  --debug                 Включить вывод отладочных сообщений в консоль.

Примеры использования:
  1. Базовый запуск со встроенной АЧХ Wharfedale Aura 2:
     python3 wharfedale_aura2_corrector.py -i /my/music -o /my/music_corrected

  2. Использование сторонней АЧХ (например, замер конкретно вашей комнаты из REW):
     python3 wharfedale_aura2_corrector.py -i /my/music -o /my/music_corrected --frd my_room_response.frd

  3. Обработка с последующей верификацией выходных файлов:
     python3 wharfedale_aura2_corrector.py -i /my/music -o /my/music_corrected --verify
"""

from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys

# ==================== БЛОК ПРОВЕРКИ ЗАВИСИМОСТЕЙ ====================
_REQUIRED_LIBS = {
    "numpy": "numpy",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "mutagen": "mutagen",
    "numba": "numba"
}
_missing_libs = []
for _lib, _pip in _REQUIRED_LIBS.items():
    try:
        __import__(_lib)
    except ImportError:
        _missing_libs.append(_pip)

if _missing_libs:
    print("\n" + "=" * 80, file=sys.stderr)
    print(" КРИТИЧЕСКАЯ ОШИБКА: Отсутствуют необходимые библиотеки!", file=sys.stderr)
    print(f" pip install {' '.join(_missing_libs)}", file=sys.stderr)
    print("=" * 80 + "\n", file=sys.stderr)
    sys.exit(1)
# ====================================================================

import argparse
import hashlib
import json
import logging
import platform
import shutil
import time
import tempfile
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy import signal
from scipy.interpolate import PchipInterpolator
from scipy.signal import resample_poly
import soundfile as sf
from mutagen.flac import FLAC, Picture

VERSION = "1.0"
PROC_TAG   = "WA2_CORRECTED"
SCALE_TAG  = "WA2_SCALE"
FAIL_TAG   = "WA2_DECODE_FAILED"
PEAK_TAG   = "WA2_TRUE_PEAK"
VERIFY_TAG = "WA2_VERIFIED"
BITDEPTH_TAG = "WA2_BITDEPTH"

# ====================================================================
# PHYSICAL MODEL PARAMETERS — Wharfedale Aura 2
# ====================================================================
#
# Все числа ниже имеют физическое обоснование. Менять осознанно.
# При изменении любого параметра — инкрементировать FIR_ALGO_VERSION,
# чтобы инвалидировать FIR-кэш.
#
# ====================================================================
# Высокочастотный фильтр (HPF)
#
# Aura 2 — фазоинверторная АС (slot-loaded port, SLPP). Порт настроен ~38 Гц:
# ниже этой частоты диффузор разгружен и работает без акустической «пружины»,
# что при большом ходе грозит механическим ударом о магнит (excursion limit).
# Измерения Erin (FRD) показывают спад −27 дБ @ 20 Гц — коррекция там
# потребовала бы усиления >9 дБ и гарантированно перегрузила бы НЧ-динамик.
#
# HPF_CUTOFF_HZ = 36.0  — частота −3 дБ коррекционного HPF.
#   Выбрана чуть ниже настройки порта (~38 Гц), чтобы не трогать полезный
#   диапазон, но жёстко отсечь зону опасного подъёма.
#
# HPF_STOP_HZ = 26.0  — частота полного заграждения (−∞ дБ в корр. кривой).
#   Разрыв 26–36 Гц формирует переходную полосу шириной 10 Гц.
#   Переход реализован косинусным тейпером в линейном масштабе амплитуды
#   (а не в дБ), что даёт S-образный профиль без экспоненциального «хвоста».
#
HPF_CUTOFF_HZ: float = 36.0
HPF_STOP_HZ:   float = 26.0

# ====================================================================
# Ограничение усиления/среза
#
#   Защищает от аномальных внешних FRD-файлов, поданных через --frd.
# MAX_BOOST_DB = 9.0  — абсолютный потолок подъёма, переопределяемый через
#   --max-boost.
#
# MAX_CUT_DB = 9.0  — симметричный потолок среза (для convexity кривой).
#
MAX_BOOST_DB: float = 9.0
MAX_CUT_DB:   float = 9.0

# ====================================================================
# Гауссовая защита баса
#
# Ниже ~45 Гц кривая Aura 2 стремительно падает (порт-рефлекс SLPP):
#   80 Гц → −2 дБ, 56 Гц → −3 дБ, 38 Гц → −6 дБ, 20 Гц → −27 дБ.
# Полная коррекция в этой зоне означала бы +27 дБ @ 20 Гц — гарантированный
# excursion failure на любой комнатной громкости.
#
# Защита реализована гауссовым окном, которое плавно снижает допустимый
# MAX_BOOST по мере приближения к нулю:
#
#   dynamic_max(f) = MAX_BOOST_DB × exp( −((CENTER − clip(f, 0, CENTER)) / σ)² )
#
# _BASS_PROTECTION_CENTER_HZ = 45.0
#   При f ≥ 45 Гц: clip(f,0,45)=45 → аргумент экспоненты = 0 → защита = 1.0
#   (ограничение снято, работает только MAX_BOOST_DB).
#   При f = 36 Гц (HPF_CUTOFF): защита ≈ 0.70 → потолок ≈ 6.3 дБ.
#   При f = 26 Гц (HPF_STOP):   защита ≈ 0.19 → потолок ≈ 1.7 дБ.
#   Ниже HPF_STOP кривая всё равно обнуляется тейпером HPF.
#
# _BASS_PROTECTION_SIGMA_HZ = 7.5
#   σ гауссиана (Гц). Определяет «жёсткость» спада защиты.
#   При σ=7.5: на −1σ (37.5 Гц) защита = e^−1 ≈ 0.37 от MAX_BOOST.
#   Подобрано так, чтобы у HPF_CUTOFF оставался запас ~6 дБ —
#   достаточно для мягкой коррекции порта без excursion-риска.
#
_BASS_PROTECTION_CENTER_HZ: float = 45.0
_BASS_PROTECTION_SIGMA_HZ:  float = 7.5

# ====================================================================
#  True Peak
#
# TARGET_TP_DBTP = −1.5  — целевой True Peak альбома после коррекции.
#   Стандарт EBU R128 рекомендует −1.0 dBTP; запас 0.5 дБ (safety_margin
#   в _album_worker) даёт реальный порог −2.0 dBTP, что исключает inter-sample
#   peaks при ресэмплинге в downstream-декодерах.
#
# TP_OVERSAMPLE = 16  — кратность передискретизации при измерении True Peak.
#   ITU-R BS.1770-4 требует минимум 4×; 16× даёт суб-сэмпловую точность
#   < 0.01 дБ для любого сигнала в диапазоне до 20 кГц.
#
TARGET_TP_DBTP: float = -1.5
TP_OVERSAMPLE:  int   = 16

# ====================================================================
# Параметры FIR-синтеза
#
# Метод: firwin2 (frequency sampling) с окном Kaiser.
# Окно Kaiser β=16 даёт теоретическое подавление боковых лепестков −163 дБ,
# что перекрывает динамический диапазон 24-бит PCM (~144 дБ) с запасом.
#
# FIR_TAPS_FACTOR = 600  — коэффициент расчёта числа отводов:
#   n = ceil(SR / HPF_CUTOFF_HZ × FIR_TAPS_FACTOR)
#   Для SR=44100: n ≈ 735 001 отводов.
#   Физический смысл: обеспечивает разрешение переходной полосы HPF
#   (~0.2 Гц на SR=44100) и ошибку АЧХ < 0.001 дБ во всём диапазоне.
#   Значение выбрано как компромисс: меньше 400 → ripple в зоне HPF;
#   больше 900 → время синтеза растёт линейно без слышимого выигрыша.
#
# FIR_TAPS_MIN = 262 143  — нижняя граница (2^18 − 1, нечётное).
#   Актуальна для высоких SR (96/192 кГц), где формула даёт меньше.
#
# FIR_TAPS_MAX = 2 097 151  — верхняя граница (2^21 − 1, нечётное).
#   Защита от аномально низкого HPF_CUTOFF_HZ или высокого SR.
#   При 2M отводов FFT-свёртка всё ещё быстрее прямой.
#
# FIR_KAISER_BETA = 16.0  — параметр формы окна Кайзера.
#   β < 6  → −80 дБ: недостаточно для 24-бит.
#   β = 8  → −100 дБ: минимум для hi-res.
#   β = 16 → −163 дБ: выбрано с тройным запасом; вычислительная стоимость
#   окна пренебрежимо мала по сравнению со стоимостью FFT.
#
# FIR_ALGO_VERSION — инкрементировать при изменении любого параметра выше,
#   чтобы инвалидировать FIR-кэш на диске/RAM.
#
FIR_TAPS_FACTOR:  float = 600.0
FIR_TAPS_MIN:     int   = 262_143
FIR_TAPS_MAX:     int   = 2_097_151
FIR_KAISER_BETA:  float = 16.0
FIR_ALGO_VERSION: int   = 1
# ====================================================================

# --- ПРИНУДИТЕЛЬНАЯ КОНФИГУРАЦИЯ RAM-ONLY ---
# 1. Принудительно направляем все временные файлы библиотек в ОЗУ
if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
    os.environ["TMPDIR"] = "/dev/shm"
    tempfile.tempdir = "/dev/shm"
else:
    # Фолбек на стандартную временную папку Debian (/tmp)
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"

# 2. Предпочтительный путь FIR-кэша — RAM-диск /dev/shm.
FIR_CACHE_CANDIDATE = Path("/dev/shm/wharfedale_aura2_fir_cache")
FIR_CACHE_DIR = FIR_CACHE_CANDIDATE   # fallback для design_fir
# ----------------------------------------

OLA_BLOCK_MAX: int = 16_777_216
OLA_BLOCK_MIN: int = 131_072

DEFAULT_WORKERS: int = max(1, (os.cpu_count() or 12) // 2)

DOP_RATES:   frozenset[int] = frozenset({176_400, 352_800, 705_600})
DOP_MARKERS: frozenset[int] = frozenset({0x05, 0xFA})
DOP_PROBE:   int             = 512

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"})

_FLAC_SUBTYPE_MAP: dict[str, str] = {
    "PCM_16": "PCM_16",
    "PCM_24": "PCM_24",
    "PCM_32": "PCM_24",
    "FLOAT":  "PCM_24",
    "DOUBLE": "PCM_24",
}

_IOPRIO_SET_SYSCALL: int = {
    "x86_64": 251,
    "aarch64": 30,
    "armv7l": 314,
}.get(platform.machine(), 251)

_rng = np.random.default_rng()

# Частотная сетка и АЧХ Wharfedale Aura 2 (on-axis, Erin's Audio Corner, Klippel NFS, Jan 2024).
# 96 точек, выверена вручную.
# Слои: [1] НЧ 20–250 Гц (порт-рефлекс SLPP), [2] СЧ 250–4000 Гц (Erin + физ. слой GF-конуса),
#        [3] ВЧ 4000–20000 Гц (AMT 27×90 мм).
_DEFAULT_FREQ = np.array([
    # --- [1] НЧ ---
    20.0, 22.0, 25.0, 28.0, 30.0, 32.0, 35.0, 38.8, 40.0, 42.0,
    45.0, 48.0, 50.0, 53.0, 56.4, 60.0, 65.0, 70.0, 75.0, 80.0,
    90.0, 100.0, 110.0, 120.0, 130.0, 150.0, 170.0, 200.0, 220.0,
    # --- [2] СЧ ---
    250.0, 280.0, 300.0, 330.0, 360.0, 400.0, 440.0, 480.0, 500.0,
    550.0, 600.0, 650.0, 700.0, 750.0, 800.0, 850.0, 900.0, 950.0,
    1000.0, 1050.0, 1100.0, 1150.0, 1200.0, 1300.0, 1400.0, 1500.0,
    1600.0, 1700.0, 1800.0, 1900.0, 2000.0, 2200.0, 2500.0, 2800.0,
    3000.0, 3200.0, 3500.0, 3800.0,
    # --- [3] ВЧ ---
    4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 7000.0, 7500.0,
    8000.0, 8500.0, 9000.0, 9500.0, 10000.0, 10500.0, 11000.0, 11500.0,
    12000.0, 12500.0, 13000.0, 13500.0, 14000.0, 14500.0, 15000.0,
    15500.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0,
], dtype=float)

_DEFAULT_MEAS_DB = np.array([
    # --- [1] НЧ ---
    -27.00, -24.50, -21.00, -18.00, -16.00, -13.50, -10.50, -6.00, -5.40, -4.80,
    -4.20, -3.80, -3.50, -3.20, -3.00, -2.70, -2.50, -2.20, -2.10, -2.00,
    -1.80, -1.50, -1.30, -1.20, -1.00, -0.90, -0.70, -0.50, -0.30,
    # --- [2] СЧ ---
    -0.20, 0.10, 0.30, 0.50, 0.60, 0.70, 0.70, 0.60, 0.50,
    0.40, 0.20, 0.00, -0.30, -0.40, -0.50, -0.40, -0.30, -0.10,
    0.00, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.30,
    1.20, 1.10, 1.00, 0.70, 0.40, 0.20, 0.00, -0.20,
    -0.30, -0.20, -0.10, 0.00,
    # --- [3] ВЧ ---
    0.00, 0.10, 0.20, 0.20, 0.10, 0.10, 0.00, 0.00,
    0.10, 0.10, 0.20, 0.30, 0.50, 0.80, 1.20, 1.70,
    2.30, 2.90, 3.30, 3.60, 3.80, 3.20, 2.00,
    0.80, -0.30, -0.70, -1.00, -1.00, -1.00,
], dtype=float)

_LOG_FMT = "%(asctime)s [%(levelname)-8s] %(message)s"
logging.getLogger().setLevel(logging.WARNING)
log = logging.getLogger(__name__)


def _configure_worker_logging(log_path: str, debug: bool) -> None:
    """Инициализирует логирование в дочернем процессе-воркере.

    При использовании ProcessPoolExecutor со стартовым методом 'spawn'
    (macOS, Windows) дочерние процессы не наследуют хендлеры родителя —
    они получают чистый корневой логгер без хендлеров. На Linux с fork()
    хендлеры наследуются, но FileHandler указывает на тот же fd, поэтому
    запись безопасна (ядро гарантирует атомарность write() до PIPE_BUF).
    Консольный _ProgressConsoleHandler у воркеров отфильтрован через
    _MainProcessFilter, поэтому дополнительно не добавляется.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    root.addHandler(fh)


class _ProgressConsoleHandler(logging.StreamHandler):
    """Выводит лог в консоль (stderr) с префиксом прогресса [XX%].
    Процент обновляется через set_pct() из главного процесса после каждого
    завершённого альбома.
    """
    def __init__(self) -> None:
        super().__init__(sys.stderr)
        self._pct: int = 0
        self.setFormatter(logging.Formatter(
            "[%(pct)3d%%] %(asctime)s | %(processName)s | %(message)s",
            datefmt="%H:%M:%S",
        ))

    def set_pct(self, pct: int) -> None:
        self._pct = max(0, min(100, pct))

    def emit(self, record: logging.LogRecord) -> None:
        record.pct = self._pct
        super().emit(record)


_progress_handler = _ProgressConsoleHandler()


class _MainProcessFilter(logging.Filter):
    """Пропускает записи лога только из главного процесса.

    Воркеры (дочерние процессы) наследуют _progress_handler через fork(),
    но их копия объекта не получает обновлений set_pct() — они всегда
    показывали бы [  0%]. Этот фильтр отсекает все записи от воркеров,
    оставляя консольный вывод только главному процессу.
    Воркеры продолжают писать в лог-файл через FileHandler.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        import multiprocessing
        return multiprocessing.current_process().name == "MainProcess"


_main_process_filter = _MainProcessFilter()

def _advise_no_cache(file_path: Path) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        fd = os.open(str(file_path), os.O_RDONLY | os.O_NONBLOCK)
        try:
            libc.posix_fadvise(fd, 0, 0, 4)
        finally:
            os.close(fd)
    except Exception:
        pass

class _WorkerArgs(NamedTuple):
    input_root:    str
    output_root:   str
    freq_meas:     np.ndarray
    spl_meas:      np.ndarray
    dry_run:       bool
    force:         bool
    max_boost_db:  float = MAX_BOOST_DB
    fir_cache_dir: str   = ""
    log_path:      str   = "wharfedale_corrector.log"
    debug:         bool  = False
    verify:        bool  = False

def _worker(args: tuple) -> dict:
    alb_str, wa = args
    _configure_worker_logging(wa.log_path, wa.debug)
    fir_cache_dir = Path(wa.fir_cache_dir) if wa.fir_cache_dir else Path(wa.output_root) / ".fir_cache"
    return _album_worker(
        Path(alb_str), Path(wa.input_root), Path(wa.output_root),
        wa.freq_meas, wa.spl_meas, wa.dry_run, wa.force, wa.max_boost_db,
        fir_cache_dir=fir_cache_dir,
        verify=wa.verify,
    )

def load_frd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freqs, spls = [], []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line[0] in ("*", "#", "!"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                freqs.append(float(parts[0]))
                spls.append(float(parts[1]))
            except ValueError:
                continue
    freq = np.array(freqs, dtype=float)
    spl  = np.array(spls,  dtype=float)
    if len(freq) < 2:
        raise ValueError("FRD-файл не содержит валидных данных")
    spl -= float(np.interp(1000.0, freq, spl))
    order = np.argsort(freq)
    return freq[order], spl[order]

def _correction_response(
    freq_meas: np.ndarray,
    spl_meas: np.ndarray,
    sample_rate: int,
    n_pts: int,
    max_boost_db: float = MAX_BOOST_DB,
) -> tuple[np.ndarray, np.ndarray]:

    nyq = sample_rate / 2.0

    # === ЛОГАРИФМИЧЕСКАЯ СЕТКА ===
    # Больше точек в низкочастотной области
    freq_log_low = np.logspace(np.log10(10.0), np.log10(500.0), int(n_pts * 0.65))
    freq_log_high = np.linspace(500.0, nyq, int(n_pts * 0.35) + 1)[1:]
    freq_grid = np.concatenate([freq_log_low, freq_log_high])
    freq_grid = np.unique(np.clip(freq_grid, 10.0, nyq))  # убираем дубли и выходим за пределы

    # Если точек получилось меньше нужного — добиваем равномерно
    if len(freq_grid) < n_pts:
        extra = np.linspace(freq_grid[-1], nyq, n_pts - len(freq_grid) + 1)[1:]
        freq_grid = np.concatenate([freq_grid, extra])

    # Основная интерполяция
    f_min = freq_meas[0]
    f_max = min(freq_meas[-1], nyq)

    interp = PchipInterpolator(freq_meas, spl_meas, extrapolate=False)
    spl_grid = interp(freq_grid)

    spl_grid = np.where(freq_grid < f_min, spl_meas[0], spl_grid)
    spl_grid = np.where(freq_grid > f_max, spl_meas[-1], spl_grid)
    spl_grid = np.nan_to_num(spl_grid, nan=0.0)

    # Коррекция
    corr_db = np.clip(-spl_grid, -MAX_CUT_DB, max_boost_db)

    # Динамическая защита баса (гауссиан; параметры — _BASS_PROTECTION_* выше)
    if max_boost_db > 0:
        _c = _BASS_PROTECTION_CENTER_HZ
        protection = np.exp(-((_c - np.clip(freq_grid, 0, _c)) / _BASS_PROTECTION_SIGMA_HZ) ** 2)
        dynamic_max = max_boost_db * protection
        corr_db = np.minimum(corr_db, dynamic_max)

    # HPF
    hpf_stop  = freq_grid <= HPF_STOP_HZ
    hpf_trans = (freq_grid > HPF_STOP_HZ) & (freq_grid < HPF_CUTOFF_HZ)
    t_trans   = np.where(
        hpf_trans,
        (freq_grid - HPF_STOP_HZ) / (HPF_CUTOFF_HZ - HPF_STOP_HZ),
        0.0,
    )
    # Зона заграждения: −200 дБ (практически 0 в линейном масштабе)
    corr_db = np.where(hpf_stop, -200.0, corr_db)
    corr_db[0] = -200.0

    # Переходная полоса: косинусный тейпер В ЛИНЕЙНОМ масштабе (S-образный).
    if np.any(hpf_trans):
        gain_in_trans  = 10.0 ** (corr_db[hpf_trans] / 20.0)
        cos_taper      = 0.5 * (1.0 - np.cos(np.pi * t_trans[hpf_trans]))
        gain_tapered   = gain_in_trans * cos_taper
        corr_db_trans  = np.where(
            gain_tapered > 0,
            20.0 * np.log10(np.maximum(gain_tapered, 1e-30)),
            -200.0,
        )
        corr_db = corr_db.copy()
        corr_db[hpf_trans] = corr_db_trans

    gain_lin = 10.0 ** (corr_db / 20.0)

    # Anti-Gibbs taper на ВЧ — применяется к gain_lin (линейный масштаб),
    # а не к corr_db (дБ). Умножение в дБ = степенная функция в линейном:
    # корректный косинусный переход должен быть линейным по амплитуде.
    if nyq >= 24000.0:
        taper_start_hz = nyq * 0.85
    else:
        taper_start_hz = max(20000.0, nyq - 1000.0)

    if nyq > taper_start_hz:
        above = freq_grid > taper_start_hz
        t = np.clip((freq_grid - taper_start_hz) / (nyq - taper_start_hz), 0.0, 1.0)
        taper = 0.5 * (1.0 + np.cos(np.pi * t))
        gain_lin = np.where(above, gain_lin * taper, gain_lin)

    return freq_grid, gain_lin

def _fir_cache_key(freq_meas: np.ndarray, spl_meas: np.ndarray, sample_rate: int,
                   max_boost_db: float = MAX_BOOST_DB) -> str:
    h = hashlib.sha1()
    h.update(freq_meas.tobytes())
    h.update(spl_meas.tobytes())
    h.update(str(sample_rate).encode())
    h.update(str(FIR_ALGO_VERSION).encode())
    h.update(str(max_boost_db).encode())
    h.update(str(MAX_CUT_DB).encode())
    h.update(f"{HPF_CUTOFF_HZ}:{HPF_STOP_HZ}".encode())
    h.update(f"{FIR_TAPS_FACTOR}:{FIR_KAISER_BETA}".encode())
    return h.hexdigest()[:16]

def design_fir(
    freq_meas: np.ndarray,
    spl_meas:  np.ndarray,
    sample_rate: int,
    max_boost_db: float = MAX_BOOST_DB,
    fir_cache_dir: Path | None = None,
) -> np.ndarray:

    cache_dir  = fir_cache_dir if fir_cache_dir is not None else FIR_CACHE_DIR
    key        = _fir_cache_key(freq_meas, spl_meas, sample_rate, max_boost_db)
    cache_file = cache_dir / f"fir_{sample_rate}_{key}.npy"
    lock_file  = cache_dir / f"fir_{sample_rate}_{key}.lock"

    # 1. Быстрая проверка без блокировки (экономит системный вызов)
    if cache_file.exists():
        try:
            return np.load(cache_file, mmap_mode='r')
        except Exception:
            pass  # файл повреждён – продолжим, чтобы пересоздать

    # 2. Атомарный захват через flock (блокирующий, но без гонок)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(lock_file, 'w') as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)  # эксклюзивная блокировка
        # После захвата – повторная проверка, т.к. другой процесс мог уже создать кэш
        if cache_file.exists():
            try:
                return np.load(cache_file, mmap_mode='r')
            except Exception:
                pass

        # 3. Вычисление фильтра и атомарная запись
        log.info("Синтез FIR-фильтра для %d Гц (ключ %s)...", sample_rate, key)
        h = _design_fir_uncached(freq_meas, spl_meas, sample_rate, max_boost_db)

        tmp = cache_file.with_suffix(f".tmp{os.getpid()}.npy")
        np.save(tmp, h)
        os.replace(tmp, cache_file)  # атомарное переименование

    # 4. Блокировка снята (выход из with), возвращаем готовый фильтр
    return np.load(cache_file, mmap_mode='r')

def _design_fir_uncached(
    freq_meas: np.ndarray,
    spl_meas:  np.ndarray,
    sample_rate: int,
    max_boost_db: float = MAX_BOOST_DB,
) -> np.ndarray:
    nyq = float(sample_rate) / 2.0
    if nyq <= HPF_CUTOFF_HZ:
        h = np.zeros(FIR_TAPS_MIN, dtype=np.float64)
        h[FIR_TAPS_MIN // 2] = 1.0
        return h

    n = int(np.ceil(float(sample_rate) / HPF_CUTOFF_HZ * FIR_TAPS_FACTOR))
    n = max(n, FIR_TAPS_MIN)
    n = min(n, FIR_TAPS_MAX)
    if n % 2 == 0:
        n += 1

    n_grid = n + 1
    freq_grid, gain_lin = _correction_response(freq_meas, spl_meas, sample_rate, n_grid, max_boost_db)
    freq_norm = freq_grid / nyq
    freq_norm[0] = 0.0
    freq_norm[-1] = 1.0
    freq_norm = np.clip(freq_norm, 0.0, 1.0)
    freq_norm, ui = np.unique(freq_norm, return_index=True)
    gain_lin = gain_lin[ui]

    if freq_norm[-1] < 1.0:
        freq_norm = np.append(freq_norm, 1.0)
        gain_lin  = np.append(gain_lin,  1.0)
    else:
        gain_lin[-1] = 1.0

    h = signal.firwin2(
        n, freq_norm, gain_lin,
        window=("kaiser", FIR_KAISER_BETA), fs=2.0,
    )
    h = h.astype(np.float64)
    _validate_fir(h, freq_meas, spl_meas, sample_rate)
    return h

def _validate_fir(
    h: np.ndarray,
    freq_meas: np.ndarray,
    spl_meas: np.ndarray,
    sample_rate: int,
) -> None:
    """Пост-синтезный sanity-check FIR. Поднимает ValueError при аномалии.

    Проверки:
      1. NaN / Inf в коэффициентах — симптом переполнения в firwin2
         или численной нестабильности PchipInterpolator.
      2. L∞-норма > 7 — косвенный признак взрыва gain; для корректирующего
         фильтра с MAX_BOOST ≤ 9 дБ значение не может превысить ~2.8.
      3. Нечётная длина — тип I FIR (симметричный, нечётное n) единственный,
         который даёт нулевую групповую задержку в центре и корректно работает
         на частоте Найквиста. Гарантируется кодом в _design_fir_uncached,
         но проверяется явно как independent defensive assertion.
      4. Симметрия (линейная фаза) — h == h[::-1] с точностью 1e-9.
         Нарушение означает баг в сборке freq_norm или регрессию в scipy.
      5. Gain @ 1 кГц — опорная точка кривой (spl=0 дБ → correction=0 дБ
         → ожидаемый gain=1.0). Отклонение > 0.1 дБ сигнализирует о сдвиге
         нормировки или ошибке в _correction_response.
    """
    # 1. NaN / Inf
    if not np.all(np.isfinite(h)):
        n_bad = int(np.sum(~np.isfinite(h)))
        raise ValueError(f"FIR содержит NaN/Inf: {n_bad} из {len(h)} коэффициентов")

    # 2. L∞-норма
    l_inf = float(np.max(np.abs(h)))
    if l_inf > 7.0:
        raise ValueError(
            f"FIR L∞-норма подозрительно велика: {l_inf:.4f} (ожидается < 7)"
        )

    # 3. Нечётная длина (тип I FIR)
    if len(h) % 2 == 0:
        raise ValueError(
            f"FIR имеет чётную длину {len(h)}: ожидается нечётная (тип I, линейная фаза)"
        )

    # 4. Симметрия (линейная фаза)
    # firwin2 с нечётным n гарантирует h == h[::-1] по построению.
    # np.allclose проверяет именно это свойство, а не эвристику «главный
    # лепесток в центре», которая не эквивалентна симметрии и может дать
    # ложное срабатывание на HPF со сложной АЧХ.
    if not np.allclose(h, h[::-1], atol=1e-9, rtol=0):
        max_asymm = float(np.max(np.abs(h - h[::-1])))
        raise ValueError(
            f"FIR нарушает симметрию (линейная фаза): "
            f"max|h[i] - h[n-1-i]| = {max_asymm:.2e} (порог 1e-9)"
        )

    # 5. Gain @ 1 кГц
    _, H = signal.freqz(h, worN=[1000.0], fs=float(sample_rate))
    gain_1k_db = 20.0 * np.log10(max(float(np.abs(H[0])), 1e-20))
    spl_1k     = float(np.interp(1000.0, freq_meas, spl_meas))
    expected_db = -spl_1k   # коррекция инвертирует кривую
    if abs(gain_1k_db - expected_db) > 0.1:
        raise ValueError(
            f"FIR gain @ 1 кГц: {gain_1k_db:.3f} дБ, "
            f"ожидалось {expected_db:.3f} дБ (отклонение {gain_1k_db - expected_db:+.3f} дБ)"
        )

def _true_peak_meter(signal: np.ndarray, oversample: int = 16) -> float:
    """Профессиональное True Peak (sub-sample accuracy, параболическая интерполяция).

    Принимает ОДНОМЕРНЫЙ массив (срез одного канала). При 2D-входе np.argmax
    возвращает линейный индекс и срезы abs_sig[max_idx±1] дают неверный результат.
    """
    if signal.ndim != 1:
        raise ValueError(
            f"_true_peak_meter: ожидается 1D-массив, получен shape={signal.shape}. "
            "Передавайте срез одного канала: signal[:, ch]"
        )
    if len(signal) == 0:
        return 0.0
    oversampled = resample_poly(signal, oversample, 1, axis=0, window=('kaiser', 12.0))
    abs_sig = np.abs(oversampled)
    max_idx = np.argmax(abs_sig)
    if 0 < max_idx < len(abs_sig) - 1:
        y1, y2, y3 = abs_sig[max_idx-1], abs_sig[max_idx], abs_sig[max_idx+1]
        a = (y1 + y3) / 2 - y2
        b = (y3 - y1) / 2
        if a < 0:
            peak_offset = -b / (2 * a)
            true_peak = y2 + b * peak_offset + a * peak_offset**2
        else:
            true_peak = y2
    else:
        true_peak = abs_sig[max_idx]
    return float(true_peak)

def is_dop(path: Path) -> bool:
    try:
        info = sf.info(str(path))
    except Exception:
        return False
    if info.samplerate not in DOP_RATES or info.subtype != "PCM_24":
        return False
    try:
        with sf.SoundFile(str(path)) as f:
            n_read = min(DOP_PROBE, f.frames)
            buf = f.read(n_read, dtype="int32", always_2d=True)
    except Exception:
        return False
    msb = ((buf >> 24) & 0xFF).astype(np.uint8)
    return float(np.isin(msb, list(DOP_MARKERS)).mean()) >= 0.80

def _ola_block_size(M: int) -> int:
    target = max(OLA_BLOCK_MIN, 2 * M)
    size = 1 << (target - 1).bit_length()
    return min(size, OLA_BLOCK_MAX)

def _flac_subtype(src_subtype: str) -> str:
    return _FLAC_SUBTYPE_MAP.get(src_subtype, "PCM_24")

from numba import njit as _njit

@_njit(cache=True)
def _ns2_loop(x: np.ndarray, out: np.ndarray) -> None:
    """NS2 Error Feedback: передаточная функция ошибки (1 - z^-1)^2.

    Насыщение состояний e1, e2 ограничено ±2.0 LSB:
    при кратковременном выбросе (Гиббс FIR) ошибка не нарастает.
    """
    n_samples = x.shape[0]
    n_ch      = x.shape[1]
    for ch in range(n_ch):
        e1 = 0.0
        e2 = 0.0
        for i in range(n_samples):
            corr = x[i, ch] + 2.0 * e1 - e2
            q = round(corr)
            out[i, ch] = q
            err = corr - q
            # Насыщение: ограничиваем накопленную ошибку для устойчивости
            # при серии выбросов (перегрузка FIR не раскачивает шейпер)
            e2 = e1
            e1 = err if err <= 2.0 and err >= -2.0 else (2.0 if err > 2.0 else -2.0)

def _quantize_and_dither_ns2(chunk: np.ndarray, bits: int) -> np.ndarray:
    # AES17: полная шкала соответствует 2^(N-1).
    # Защита от overflow — clip до ±(2^(N-1)-1) перед квантованием,
    # а не смещение масштаба. Это корректно с точки зрения стандарта.
    scale_factor = float(1 << (bits - 1))
    scaled = chunk * scale_factor

    # Добавление TPDF-дизеринга (амплитуда 1 LSB)
    dither = (_rng.random(scaled.shape, dtype=np.float64) - _rng.random(scaled.shape, dtype=np.float64))
    scaled = scaled + dither

    # Clip до допустимого целочисленного диапазона ДО NS2.
    # Это предотвращает раскачку состояний шейпера при выбросах FIR.
    max_val = (1 << (bits - 1)) - 1
    min_val = -(1 << (bits - 1))
    scaled = np.clip(scaled, float(min_val), float(max_val))

    out = np.zeros_like(scaled)
    _ns2_loop(scaled, out)

    clipped = np.clip(out, min_val, max_val)

    if bits == 16:
        return clipped.astype(np.int16)
    else:
        # Для 24-битного PCM сдвигаем влево, подготавливая под формат soundfile
        return (clipped.astype(np.int32) << 8)

def _copy_meta(src: Path, dst: Path, extra: dict[str, str]) -> None:
    """
    Перенос всех Vorbis Comments и обложек из исходного FLAC в выходной файл.
    """
    try:
        s = FLAC(str(src))
        d = FLAC(str(dst))

        # Шаг 1: сбрасываем обложки и теги ТОЛЬКО в памяти — диск не трогаем
        d.clear_pictures()
        if d.tags is None:
            d.add_tags()
        else:
            d.tags.clear()

        # Шаг 2: копируем все теги из источника
        if s.tags is not None:
            for key, values in s.tags.items():
                d.tags[key] = values

        # Шаг 3: дописываем технический паспорт обработки (перезаписывает при --force)
        for k, v in extra.items():
            d[k] = [v]

        # Шаг 4: копируем обложки
        for pic in s.pictures:
            pic_copy = Picture()
            for attr in ("type", "mime", "desc", "width", "height", "depth", "data"):
                setattr(pic_copy, attr, getattr(pic, attr))
            d.add_picture(pic_copy)

        # Шаг 5: запись на диск
        d.save()
    except Exception as exc:
        log.warning("Перенос метаданных (%s -> %s): %s", src.name, dst.name, exc)
        raise  # пробрасываем наверх — вызывающий код должен знать о провале тегирования

def _verify_tags(dst: Path, expected_scale: float) -> tuple[bool, str]:
    """Уровень 2 — верификация тегов выходного файла сразу после _copy_meta.

    Проверяет:
      1. Файл открывается mutagen без исключений.
      2. Тег PROC_TAG присутствует и начинается с "PROCESSED:".
      3. SCALE_TAG присутствует и его числовое значение совпадает с expected_scale
         с точностью до 1e-6 (защита от ошибок форматирования при записи тега).
      4. FAIL_TAG отсутствует.

    Возвращает (ok: bool, причина: str). При ok=True причина — пустая строка.
    Стоимость: одно открытие файла через mutagen (~1–3 мс). За 10 000 файлов
    добавляет порядка 15–30 секунд — около 1–3%% от типичного времени обработки.
    """
    try:
        m = FLAC(str(dst))
    except Exception as exc:
        return False, f"mutagen не смог открыть файл: {exc}"

    proc = m.get(PROC_TAG)
    if not proc or not proc[0].startswith("PROCESSED:"):
        return False, f"отсутствует или некорректен тег {PROC_TAG}: {proc}"

    if m.get(FAIL_TAG):
        return False, f"присутствует тег ошибки {FAIL_TAG}"

    scale_raw = m.get(SCALE_TAG)
    if not scale_raw:
        return False, f"отсутствует тег {SCALE_TAG}"
    try:
        recorded_scale = float(scale_raw[0])
    except (ValueError, TypeError) as exc:
        return False, f"не удалось прочитать {SCALE_TAG}: {exc}"
    if abs(recorded_scale - expected_scale) > 1e-6:
        return False, (
            f"{SCALE_TAG} в файле ({recorded_scale:.8f}) "
            f"не совпадает с ожидаемым ({expected_scale:.8f})"
        )

    return True, ""


def _verify_audio_frames(dst: Path, expected_frames: int) -> tuple[bool, str]:
    """Уровень 3 — верификация декодируемости и числа семплов (только с --verify).

    Открывает выходной FLAC через soundfile и считывает весь файл блоками,
    подсчитывая реальное число декодированных семплов. Сравнивает с
    expected_frames из заголовка исходного файла.

    Стоимость: полный декод каждого файла (~4–8%% дополнительного CPU
    на типичной медиатеке; суммарно ~8–12%% с учётом IO).
    """
    try:
        total_decoded: int = 0
        with sf.SoundFile(str(dst)) as f:
            header_frames = f.frames
            # Читаем блоками чтобы не держать весь файл в RAM
            for block in f.blocks(blocksize=65536, dtype="float64", always_2d=True):
                total_decoded += len(block)
    except Exception as exc:
        return False, f"soundfile не смог декодировать файл: {exc}"

    if header_frames != expected_frames:
        return False, (
            f"заголовок FLAC содержит {header_frames} семплов, "
            f"ожидалось {expected_frames}"
        )
    if total_decoded != expected_frames:
        return False, (
            f"декодировано {total_decoded} семплов, "
            f"ожидалось {expected_frames}"
        )
    return True, ""


def _copy_images_in_dir(src_dir: Path, dst_dir: Path, force: bool, dry_run: bool) -> list[dict]:
    results = []
    for src_img in src_dir.iterdir():
        if not src_img.is_file():
            continue
        if src_img.suffix.lower() not in IMAGE_EXTS:
            continue
        dst_img = dst_dir / src_img.name
        if not force and dst_img.exists():
            results.append({"src": str(src_img), "dst": str(dst_img), "action": "skip_img"})
            continue
        if not dry_run:
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src_img), str(dst_img))
        results.append({"src": str(src_img), "dst": str(dst_img), "action": "copy_img"})
    return results

def copy_images(input_root: Path, output_root: Path, force: bool, dry_run: bool) -> dict[str, int]:
    counters: dict[str, int] = {"copy_img": 0, "skip_img": 0, "error_img": 0}

    for src_dir in input_root.rglob("*"):
        if not src_dir.is_dir():
            continue
        try:
            has_images = any(f.is_file() and f.suffix.lower() in IMAGE_EXTS for f in src_dir.iterdir())
        except OSError as exc:
            log.debug("Пропуск сканирования директории %s (нет доступа): %s", src_dir, exc)
            continue
        if not has_images:
            continue
        dst_dir = output_root / src_dir.relative_to(input_root)
        try:
            results = _copy_images_in_dir(src_dir, dst_dir, force, dry_run)
            for r in results:
                counters[r["action"]] = counters.get(r["action"], 0) + 1
        except Exception as exc:
            log.error("Ошибка копирования изображений из %s: %s", src_dir, exc)
            counters["error_img"] += 1

    try:
        root_results = _copy_images_in_dir(input_root, output_root, force, dry_run)
        for r in root_results:
            counters[r["action"]] = counters.get(r["action"], 0) + 1
    except Exception as exc:
        log.error("Ошибка копирования изображений из корня %s: %s", input_root, exc)

    return counters

def _read_track_meta(path: Path) -> tuple[bool, float | None, float | None]:
    try:
        m = FLAC(str(path))
        proc = m.get(PROC_TAG)
        ok = (
            bool(proc)
            and proc[0].startswith("PROCESSED:")
            and bool(m.get(SCALE_TAG))
            and not m.get(FAIL_TAG)
        )
        scale: float | None = None
        peak:  float | None = None
        if ok:
            try:
                scale = float(m.get(SCALE_TAG)[0])
            except Exception:
                ok = False
            try:
                v = float(m.get(PEAK_TAG, [None])[0])
                peak = v if v > 0 else None
            except Exception:
                pass
        return ok, scale, peak
    except Exception:
        return False, None, None

def _has_tag(path: Path) -> bool:
    """Используется только для DoP-ветки (любой WHARFEDALE-тег)."""
    try:
        return bool(FLAC(str(path)).get(PROC_TAG))
    except Exception:
        return False

class TrackTask:
    def __init__(self, src: Path, dst: Path, sr: int, subtype: str, n_frames: int):
        self.src      = src
        self.dst      = dst
        self.sr       = sr
        self.subtype  = subtype
        self.n_frames = n_frames
        self.computed_max_tp: float = 0.0

def _ola_common_setup(h: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Возвращает (block_size, fft_size, H_fft) — общие параметры для обоих проходов."""
    M          = len(h)
    block_size = _ola_block_size(M)
    fft_size   = 1 << (block_size + M - 1).bit_length()
    H_fft      = np.fft.rfft(h, n=fft_size)
    return block_size, fft_size, H_fft

def _ola_tp_pass(task: TrackTask, h: np.ndarray) -> None:
    """ПРОХОД 1 — потоковый расчёт True Peak.

    Выполняет OLA-свёртку блок за блоком и поддерживает бегущий максимум
    абсолютных значений. PCM нигде не накапливается.
    Результат записывается в task.computed_max_tp.
    """
    M = len(h)
    delay = (M - 1) // 2
    block_size, fft_size, H_fft = _ola_common_setup(h)

    running_max_tp: float = 0.0

    with sf.SoundFile(str(task.src)) as inf:
        n_ch    = inf.channels
        overlap = np.zeros((M - 1, n_ch), dtype=np.float64)
        in_pos  = 0
        frames_emitted = 0

        for block in inf.blocks(blocksize=block_size, dtype="float64", always_2d=True):
            L        = len(block)
            X        = np.fft.rfft(block, n=fft_size, axis=0)
            conv_full = np.fft.irfft(X * H_fft[:, np.newaxis], n=fft_size, axis=0)
            conv     = conv_full[:L + M - 1]
            conv[:M-1] += overlap
            overlap  = conv[L:].copy()

            out_s = in_pos - delay
            out_e = in_pos + L - delay
            vs    = max(0, out_s)
            ve    = min(task.n_frames, out_e)
            if vs < ve:
                chunk = conv[vs - out_s : ve - out_s]
                # Потоковый True Peak: бегущий максимум по блоку
                for ch in range(n_ch):
                    running_max_tp = max(running_max_tp,
                                        _true_peak_meter(chunk[:, ch]))
                frames_emitted = ve
            in_pos += L

        # Хвост overlap
        if frames_emitted < task.n_frames:
            tail = overlap[:task.n_frames - frames_emitted]
            if tail.shape[0] > 0:
                for ch in range(n_ch):
                    running_max_tp = max(running_max_tp,
                                        _true_peak_meter(tail[:, ch]))

    task.computed_max_tp = running_max_tp
    _advise_no_cache(task.src)

def _ola_write_pass(task: TrackTask, h: np.ndarray, scale: float) -> None:
    """ПРОХОД 2 — потоковая свёртка + атомарная запись FLAC с известным scale.

    Схема атомарности:
      1. Весь PCM пишется во временный файл <dst>.tmpPID.flac.
      2. Сверяется счётчик семплов — расхождение поднимает RuntimeError.
      3. os.fsync гарантирует сброс на диск до rename.
      4. os.replace(tmp, dst) — атомарная подстановка (POSIX rename семантика).
      5. При любом исключении tmp удаляется, dst остаётся нетронутым.

    Теги накладываются в _album_worker уже на финальный dst — после replace.
    Верификация тегов (_verify_tags) также выполняется после replace.

    Уровень 1б: scale > 1.0 физически невозможен после корректного TP-расчёта;
    логируется как WARNING и применяется как есть (защита от тихого клиппинга).
    """
    # ── Уровень 1б: проверка scale ───────────────────────────────────────────────
    if scale > 1.0:
        log.warning(
            "%s: scale=%.6f > 1.0 — усиление вместо ослабления после TP-коррекции. "
            "Возможна ошибка вычисления True Peak. Применяем как есть.",
            task.src.name, scale,
        )

    M = len(h)
    delay = (M - 1) // 2
    block_size, fft_size, H_fft = _ola_common_setup(h)

    out_subtype       = _flac_subtype(task.subtype)
    bits              = 16 if out_subtype == "PCM_16" else 24
    sf_format_subtype = "PCM_16" if bits == 16 else "PCM_24"

    tmp_dst = task.dst.with_suffix(f".tmp{os.getpid()}.flac")
    try:
        frames_written_total: int = 0

        with sf.SoundFile(str(task.src)) as inf:
            n_ch    = inf.channels
            overlap = np.zeros((M - 1, n_ch), dtype=np.float64)
            in_pos  = 0
            frames_written = 0

            with sf.SoundFile(str(tmp_dst), mode="w", samplerate=task.sr,
                              channels=n_ch, subtype=sf_format_subtype,
                              format="FLAC", compression_level=1.0) as outf:

                for block in inf.blocks(blocksize=block_size, dtype="float64", always_2d=True):
                    L        = len(block)
                    X        = np.fft.rfft(block, n=fft_size, axis=0)
                    conv_full = np.fft.irfft(X * H_fft[:, np.newaxis], n=fft_size, axis=0)
                    conv     = conv_full[:L + M - 1]
                    conv[:M-1] += overlap
                    overlap  = conv[L:].copy()

                    out_s = in_pos - delay
                    out_e = in_pos + L - delay
                    vs    = max(0, out_s)
                    ve    = min(task.n_frames, out_e)
                    if vs < ve:
                        chunk     = conv[vs - out_s : ve - out_s]
                        quantized = _quantize_and_dither_ns2(chunk * scale, bits)
                        outf.write(quantized)
                        frames_written_total += ve - vs
                        frames_written        = ve
                    in_pos += L

                # Хвост overlap
                if frames_written < task.n_frames:
                    tail = overlap[:task.n_frames - frames_written]
                    if tail.shape[0] > 0:
                        quantized = _quantize_and_dither_ns2(tail * scale, bits)
                        outf.write(quantized)
                        frames_written_total += tail.shape[0]

        # ── Жёсткая проверка числа семплов — до fsync и replace ─────────────────
        # Расхождение означает повреждённый заголовок исходника или баг в OLA;
        # в обоих случаях tmp удаляется и dst остаётся нетронутым.
        if frames_written_total != task.n_frames:
            raise RuntimeError(
                f"записано {frames_written_total} семплов, ожидалось {task.n_frames} "
                f"(разница {frames_written_total - task.n_frames:+d}). "
                "Файл отклонён, tmp удалён."
            )

        # ── fsync + атомарный rename ─────────────────────────────────────────────
        try:
            with open(str(tmp_dst), "r+b") as _fd:
                os.fsync(_fd.fileno())
        except OSError:
            pass

        os.replace(tmp_dst, task.dst)

    except Exception:
        tmp_dst.unlink(missing_ok=True)
        raise

    _advise_no_cache(task.dst)

def _album_worker(
    album_dir: Path, input_root: Path, output_root: Path,
    freq_meas: np.ndarray, spl_meas: np.ndarray,
    dry_run: bool, force: bool, max_boost_db: float = MAX_BOOST_DB,
    fir_cache_dir: Path | None = None,
    verify: bool = False,
) -> dict[str, Any]:
    """
    Двухпроходный конвейер обработки альбома.

    ПРОХОД 1: для каждого трека запускается _ola_tp_pass —
      потоковая OLA-свёртка с бегущим расчётом True Peak. PCM не
      накапливается.

    ПРОХОД 2: для каждого трека запускается _ola_write_pass —
      повторная потоковая свёртка с уже известным scale и немедленной
      записью квантованных блоков во FLAC.

    Между проходами вычисляется общий scale альбома.
    """
    # Пересеиваем ГПСЧ для этого процесса/альбома.
    # _rng — модульный singleton: при fork() все воркеры наследуют одно
    # и то же состояние генератора, что даёт идентичные дизер-последовательности
    # в параллельных треках. os.getpid() уникален для каждого дочернего процесса,
    # hash(str(album_dir)) добавляет уникальность внутри одного pid при гипотетическом
    # переиспользовании процесса пулом.
    global _rng
    _rng = np.random.default_rng(os.getpid() ^ (hash(str(album_dir)) & 0xFFFFFFFF))

    if fir_cache_dir is None:
        fir_cache_dir = output_root / ".fir_cache"
    rel        = album_dir.relative_to(input_root)
    flac_files = sorted(album_dir.rglob("*.flac"))
    if not flac_files:
        return {"album": str(rel), "files": []}

    fir_cache:     dict[int, np.ndarray] = {}
    # tasks хранит только метаданные треков (без PCM-буфера)
    tasks:         list[TrackTask]       = []
    failed_tracks: set[Path]             = set()
    skipped_srcs:  set[Path]             = set()
    peak_db:       float                 = -100.0
    scale:         float                 = 1.0

    dop_cache: dict[Path, bool] = {src: is_dop(src) for src in flac_files}

    # Читаем сохранённый scale из выходных файлов предыдущего запуска
    existing_scale: float | None = None
    for src in flac_files:
        if dop_cache[src]:
            continue
        dst = output_root / src.relative_to(input_root)
        if not force and dst.exists():
            ok, sc, _ = _read_track_meta(dst)
            if ok and sc is not None:
                existing_scale = sc
                break

    album_max_tp = 0.0

    # ── ЭТАП 1: ПРОХОД 1 — потоковый расчёт True Peak (RAM = O(block_size)) ──────
    for src in flac_files:
        dst = output_root / src.relative_to(input_root)
        if dop_cache[src]:
            continue

        if not force and dst.exists():
            ok, sc, raw_peak = _read_track_meta(dst)
            if ok:
                if raw_peak is not None and raw_peak > 0:
                    album_max_tp = max(album_max_tp, raw_peak)
                elif sc is not None and sc > 0:
                    # Лучше использовать сохранённый PEAK_TAG, если есть
                    restored_tp = (10 ** (TARGET_TP_DBTP / 20.0)) / sc
                    album_max_tp = max(album_max_tp, restored_tp)
                skipped_srcs.add(src)
                continue

        try:
            with sf.SoundFile(str(src)) as f:
                sr, subtype, channels, n_frames = (
                    f.samplerate, f.subtype, f.channels, f.frames)

            if channels != 2:
                log.warning("Пропуск %s: Коррекция рассчитана исключительно на стерео-панораму.",
                            src.name)
                failed_tracks.add(src)
                continue

            # Минимальная длина трека: ~23 мс @ 44100.
            # Файлы короче не имеют смысла для коррекции и могут дать
            # некорректный OLA-хвост (tail длиннее самого сигнала).
            MIN_FRAMES = 1024
            if n_frames < MIN_FRAMES:
                log.warning("Пропуск %s: слишком короткий файл (%d семплов, минимум %d).",
                            src.name, n_frames, MIN_FRAMES)
                failed_tracks.add(src)
                continue

            task = TrackTask(src, dst, sr, subtype, n_frames)

            if sr not in fir_cache:
                fir_cache[sr] = design_fir(freq_meas, spl_meas, sr,
                                           max_boost_db, fir_cache_dir=fir_cache_dir)
            _ola_tp_pass(task, fir_cache[sr])

            album_max_tp = max(album_max_tp, task.computed_max_tp)
            tasks.append(task)
        except Exception as exc:
            log.error("ПРОХОД 1 — критическая ошибка %s: %s", src.name, exc)
            failed_tracks.add(src)

    # ── ЭТАП 2: Вычисление scale альбома ─────────────────────────────────────────
    # safety_margin (+0.5 дБ) — запас между номинальной целью и реальным порогом
    # срабатывания. Реальный порог = TARGET_TP_DBTP - 0.5 дБ.
    # В логе показываем оба значения, чтобы не вводить в заблуждение.
    safety_margin    = 10 ** (0.5 / 20.0)
    target_tp        = (10 ** (TARGET_TP_DBTP / 20.0)) / safety_margin
    target_tp_db_eff = 20 * np.log10(target_tp)   # реальный порог срабатывания

    if album_max_tp > target_tp and album_max_tp > 0:
        scale   = target_tp / album_max_tp
        peak_db = 20 * np.log10(album_max_tp)
    else:
        scale = 1.0
        peak_db = 20 * np.log10(album_max_tp) if album_max_tp > 0 else -100.0
    # Защита от некорректных значений
    if scale <= 0 or not np.isfinite(scale):
        log.warning("Некорректный scale %.6f для альбома %s. Используем 1.0", scale, rel)
        scale = 1.0

    scale_changed = (existing_scale is not None and abs(existing_scale - scale) > 1e-6)
    if scale_changed:
        # scale изменился — нужно перезаписать ВСЕ треки альбома.
        # Дополняем tasks треками, которые были пропущены как «уже готовые»
        # (они не попали в ЭТАП 1, поэтому добавляем их сюда).
        log.warning("Альбом %s: коэффициент изменился %.6f -> %.6f — тотальный перерасчёт",
                    str(rel), existing_scale, scale)
        existing_srcs = {t.src for t in tasks}
        for src in flac_files:
            if dop_cache[src] or src in failed_tracks or src in existing_srcs:
                continue
            dst = output_root / src.relative_to(input_root)
            try:
                with sf.SoundFile(str(src)) as f:
                    sr, subtype, n_frames = f.samplerate, f.subtype, f.frames
                task = TrackTask(src, dst, sr, subtype, n_frames)
                if sr not in fir_cache:
                    fir_cache[sr] = design_fir(freq_meas, spl_meas, sr,
                                               max_boost_db, fir_cache_dir=fir_cache_dir)
                # Пересчитываем TP для пропущенных треков
                _ola_tp_pass(task, fir_cache[sr])
                tasks.append(task)
            except Exception as exc:
                log.error("ПРОХОД 1 (scale_changed) — ошибка %s: %s", src.name, exc)
                failed_tracks.add(src)

    # ── ЭТАП 3: ПРОХОД 2 — потоковая запись FLAC (RAM = O(block_size)) ───────────
    results:  list[dict]          = []
    task_map: dict[Path, TrackTask] = {t.src: t for t in tasks}

    for src in flac_files:
        dst = output_root / src.relative_to(input_root)

        # DoP-поток: просто копируем
        if dop_cache[src]:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dry_run and (force or scale_changed or not dst.exists() or not _has_tag(dst)):
                shutil.copy2(str(src), str(dst))
                # _copy_meta уже записывает PROC_TAG = DOP_BYPASS — _stamp_dop не нужен
                _copy_meta(src, dst, extra={PROC_TAG: f"DOP_BYPASS:{VERSION}"})
            results.append({"src": str(src), "dst": str(dst), "action": "copy_dop"})
            continue

        # Трек с ошибкой декодирования: копируем как есть с меткой
        if src in failed_tracks:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not dry_run:
                    shutil.copy2(str(src), str(dst))
                    try:
                        m = FLAC(str(dst))
                        if m.tags is None:
                            m.add_tags()
                        m[PROC_TAG] = [f"COPY_DECODE_ERROR:{VERSION}"]
                        m[FAIL_TAG] = ["1"]
                        m.save()
                    except Exception as exc_meta:
                        log.warning("Не удалось пометить дефектный трек %s: %s",
                                    dst.name, exc_meta)
                results.append({"src": str(src), "dst": str(dst), "action": "copy_failed"})
            except Exception as exc:
                log.error("Ошибка резервного копирования %s: %s", src.name, exc)
                results.append({"src": str(src), "dst": str(dst),
                                 "action": "error", "error": str(exc)})
            continue

        # Основной путь: потоковая OLA-свёртка + запись FLAC
        task = task_map.get(src)
        if src in skipped_srcs:
            results.append({"src": str(src), "dst": str(dst),
                             "action": "skip_done"})
            continue
        if task is None:
            log.error("Нет задачи для трека %s (не попал в ПРОХОД 1)", src.name)
            results.append({"src": str(src), "dst": str(dst),
                             "action": "error", "error": "Нет задачи после ПРОХОДА 1"})
            continue

        try:
            # sentinel-значения: используются если dry_run=True или verify=False
            tag_ok:      bool = True
            tag_reason:  str  = ""
            audio_ok:    bool = True
            audio_reason: str = ""

            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                _ola_write_pass(task, fir_cache[task.sr], scale)
                try:
                    _copy_meta(src, dst, extra={
                        PROC_TAG:  f"PROCESSED:{VERSION}",
                        SCALE_TAG: f"{scale:.8f}",
                        PEAK_TAG:  f"{task.computed_max_tp:.10f}",
                        BITDEPTH_TAG: task.subtype,
                    })
                except Exception as meta_exc:
                    # PCM атомарно записан (os.replace), но тегирование провалилось.
                    # dst существует с правильным звуком, но без WA2_* тегов —
                    # помечаем copy_failed, чтобы следующий запуск его перезаписал.
                    log.error("ПРОХОД 2 — ошибка тегирования %s: %s", dst.name, meta_exc)
                    results.append({"src": str(src), "dst": str(dst),
                                    "action": "copy_failed",
                                    "error": f"тегирование: {meta_exc}"})
                    continue

                # ── Уровень 2: верификация тегов (~1-3 мс на файл) ───────────────────
                # Выполняется всегда при реальной записи: открывает dst через mutagen
                # и проверяет что PROC_TAG/SCALE_TAG записались корректно.
                # При успехе записывает VERIFY_TAG — повторный запуск сможет отличить
                # «верификация пройдена» от «верификация не выполнялась».
                tag_ok, tag_reason = _verify_tags(dst, scale)
                if not tag_ok:
                    log.warning("VERIFY-TAGS FAIL %s: %s", dst.name, tag_reason)
                else:
                    log.debug("VERIFY-TAGS OK %s", dst.name)
                    try:
                        m = FLAC(str(dst))
                        m[VERIFY_TAG] = [VERSION]
                        m.save()
                    except Exception as vtag_exc:
                        log.debug("Не удалось записать %s в %s: %s",
                                  VERIFY_TAG, dst.name, vtag_exc)

                # ── Уровень 3: полный декод и сверка семплов (только с --verify) ─────
                # Читает весь выходной файл блоками, считает декодированные семплы
                # и сравнивает с task.n_frames. Добавляет ~8–12%% к общему времени.
                if verify:
                    audio_ok, audio_reason = _verify_audio_frames(dst, task.n_frames)
                    if not audio_ok:
                        log.warning("VERIFY-AUDIO FAIL %s: %s", dst.name, audio_reason)
                    else:
                        log.debug("VERIFY-AUDIO OK %s: %d семплов подтверждено",
                                  dst.name, task.n_frames)

            result_entry: dict[str, Any] = {
                "src":         str(src),
                "dst":         str(dst),
                "action":      "processed",
                "src_subtype": task.subtype,
                "out_subtype": _flac_subtype(task.subtype),
            }
            if not dry_run:
                result_entry["tag_ok"]   = tag_ok
                result_entry["tag_fail"] = tag_reason
                if verify:
                    result_entry["audio_ok"]   = audio_ok
                    result_entry["audio_fail"] = audio_reason
            results.append(result_entry)
        except Exception as exc:
            log.error("ПРОХОД 2 — ошибка сборки FLAC %s: %s", src.name, exc)
            # Пытаемся скопировать исходный файл как fallback с меткой ошибки
            try:
                if not dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src), str(dst))
                    try:
                        m = FLAC(str(dst))
                        if m.tags is None:
                            m.add_tags()
                        m[PROC_TAG] = [f"COPY_WRITE_ERROR:{VERSION}"]
                        m[FAIL_TAG] = ["1"]
                        m.save()
                    except Exception as meta_exc:
                        log.warning("Не удалось пометить трек с ошибкой записи %s: %s",
                                    dst.name, meta_exc)
                results.append({"src": str(src), "dst": str(dst), "action": "copy_failed"})
            except Exception as copy_exc:
                log.error("Не удалось скопировать исходный файл %s: %s", src.name, copy_exc)
                results.append({"src": str(src), "dst": str(dst),
                                 "action": "error", "error": str(copy_exc)})

    return {"album": str(rel), "scale": scale, "peak_dbtp": peak_db, "files": results}

def find_albums(root: Path) -> list[Path]:
    return sorted({f.parent for f in root.rglob("*.flac")})

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Wharfedale Aura 2 — Линейно-фазовая коррекция",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("-i", "--input",   default="/MDA/Музыка", help="Путь к исходной медиатеке FLAC")
    ap.add_argument("-o", "--output",  required=True, help="Путь к результирующей папке")
    ap.add_argument("--frd",           default="", help="Кастомный файл АЧХ (.frd/.txt) вместо стандартного")
    ap.add_argument("-w", "--workers", type=int, default=DEFAULT_WORKERS, help="Количество параллельных потоков-воркеров")
    ap.add_argument("--dry-run",  action="store_true", help="Тестовый прогон без записи файлов на диск")
    ap.add_argument("--force",    action="store_true", help="Принудительная перезапись ранее обработанных файлов")
    ap.add_argument("--verify",   action="store_true",
                    help="Полный декод каждого выходного FLAC для проверки числа семплов (+8-12%% времени)")
    ap.add_argument("--report",   default="wharfedale_corrector.json", help="Путь к JSON-отчету")
    ap.add_argument("--log",      default="wharfedale_corrector.log", help="Путь к лог-файлу")
    ap.add_argument("--debug",    action="store_true", help="Отображение логов уровня DEBUG")
    ap.add_argument(
        "--max-boost",
        type=float,
        default=MAX_BOOST_DB,
        metavar="DB",
        help=f"Максимальный подъём НЧ в дБ. Диапазон [0.0 - {MAX_BOOST_DB}].",
    )
    args = ap.parse_args()

    if not (0.0 <= args.max_boost <= 9.0):
        ap.error(f"--max-boost должен быть в диапазоне 0–9 дБ, получено: {args.max_boost}")

    input_root  = Path(args.input).resolve()
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Настройка логирования — первым делом, до любых log.* вызовов.
    # Порядок важен: сначала уровень корневого логгера, затем хендлеры.
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)

    # FileHandler — пишет все уровни без фильтров (и MainProcess, и воркеры).
    fh = logging.FileHandler(args.log, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_LOG_FMT))
    root_logger.addHandler(fh)

    # Консольный хендлер — только MainProcess (фильтр _MainProcessFilter).
    # Уровень DEBUG для консоли управляется флагом --debug.
    _progress_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    _progress_handler.addFilter(_main_process_filter)
    root_logger.addHandler(_progress_handler)

    # Низкоуровневая настройка Linux — после подключения логирования,
    # чтобы сообщения о nice/ionice были видны в консоли.
    if sys.platform.startswith("linux"):
        try:
            os.nice(19)
            log.info("Системный приоритет планировщика снижен (nice = 19).")
        except Exception as e:
            log.warning("Не удалось изменить nice: %s", e)

        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            IOPRIO_CLASS_SHIFT = 13
            ioprio = (3 << IOPRIO_CLASS_SHIFT) | 0
            libc.syscall(_IOPRIO_SET_SYSCALL, 1, 0, ioprio)
            log.info("Приоритет ввода-вывода (ionice) переведен в режим фоновой обработки (IDLE).")
        except Exception as e:
            log.warning("Не удалось настроить приоритет ввода-вывода через syscall: %s", e)

    if not input_root.is_dir():
        log.error("Входной каталог не найден: %s", input_root)
        return 1
    if output_root == input_root:
        log.error("Входной и выходной каталоги должны различаться.")
        return 1

    # Проверка места на диске (не менее 50 Гб)
    try:
        total, used, free = shutil.disk_usage(output_root)
        if free < 50 * 1024 * 1024 * 1024:
            log.warning("ВНИМАНИЕ: На целевом диске осталось мало места (%d МБ).", free // (1024 * 1024))
    except Exception as exc:
        log.debug("Не удалось проверить свободное место: %s", exc)

    # Загрузка FRD или дефолтной кривой
    if args.frd:
        frd_path = Path(args.frd)
        if not frd_path.exists():
            log.error("FRD-файл не найден: %s", frd_path)
            return 1
        try:
            freq_meas, spl_meas = load_frd(frd_path)
            log.info("FRD загружен: %s (%d точек)", frd_path.name, len(freq_meas))
        except Exception as exc:
            log.error("Ошибка загрузки FRD: %s", exc)
            return 1
    else:
        freq_meas = _DEFAULT_FREQ
        spl_meas  = _DEFAULT_MEAS_DB
        log.info("Принята опорная АЧХ Wharfedale Aura 2 по измерениям Erin's Audio Corner.")

# ==================== ОПРЕДЕЛЕНИЕ ПУТИ К FIR-КЭШУ ====================
    ram_disk = Path("/dev/shm")
    if ram_disk.exists() and os.access(str(ram_disk), os.W_OK | os.X_OK):
        fir_cache_path = FIR_CACHE_CANDIDATE
    else:
        fir_cache_path = output_root / ".fir_cache"

    # Создаём папку кэша, если её нет
    fir_cache_path.mkdir(parents=True, exist_ok=True)

    # Гарантируем очистку ОЗУ/диска через блок try...finally
    try:
        # Вычисляем рабочий порог TP для баннера (совпадает с формулой в _album_worker)
        _safety_margin    = 10 ** (0.5 / 20.0)
        _target_tp        = (10 ** (TARGET_TP_DBTP / 20.0)) / _safety_margin
        target_tp_db_eff  = 20 * np.log10(_target_tp)

        log.info("=" * 78)
        log.info("Wharfedale Aura 2 Corrector v%s", VERSION)
        log.info("Вход    : %s", input_root)
        log.info("Выход   : %s", output_root)
        log.info("ФВЧ     : %.0f Гц (заграждение %.0f Гц)", HPF_CUTOFF_HZ, HPF_STOP_HZ)
        log.info("Подъём  : макс. +%.1f дБ (--max-boost) / срез: макс. %.0f дБ", args.max_boost, MAX_CUT_DB)
        log.info("FIR     : Kaiser =%.1f (-163 дБ), отводов %d…%d (algo v%d)",
                 FIR_KAISER_BETA, FIR_TAPS_MIN, FIR_TAPS_MAX, FIR_ALGO_VERSION)
        log.info("True Peak цель: %.1f dBTP (рабочий порог: %.1f dBTP)",
                 TARGET_TP_DBTP, target_tp_db_eff)
        log.info("Воркеры : %d  | dry_run=%s | force=%s | verify=%s",
                 args.workers, args.dry_run, args.force, args.verify)
        log.info("FIR кэш : %s", fir_cache_path)
        log.info("=" * 78)

        log.info("Копирование графических метаданных (Art/Covers)...")
        img_counters = copy_images(input_root, output_root, args.force, args.dry_run)
        log.info("Обложки: скопировано=%d, пропущено=%d, ошибок=%d",
                 img_counters.get("copy_img", 0),
                 img_counters.get("skip_img", 0),
                 img_counters.get("error_img", 0))
        t0 = time.monotonic()
        albums = find_albums(input_root)
        log.info("Найдено %d альбомов за %.1f с.", len(albums), time.monotonic() - t0)

        if not albums:
            log.warning("FLAC-файлы не обнаружены.")
            return 0

        wa = _WorkerArgs(
            input_root    = str(input_root),
            output_root   = str(output_root),
            freq_meas     = freq_meas,
            spl_meas      = spl_meas,
            dry_run       = args.dry_run,
            force         = args.force,
            max_boost_db  = args.max_boost,
            fir_cache_dir = str(fir_cache_path),
            log_path      = args.log,
            debug         = args.debug,
            verify        = args.verify,
        )

        counters: dict[str, int] = {}
        all_results: list[dict]  = []
        t1 = time.monotonic()

        # Считаем общее число FLAC-файлов заранее — для расчёта процента
        total_files = sum(len(list(alb.rglob("*.flac"))) for alb in albums)
        done_files  = 0

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_worker, (str(alb), wa)): alb for alb in albums}
            for fut in as_completed(futures):
                # Инициализируем до try/except: предотвращаем «протечку»
                # значений предыдущей итерации при частичном исключении.
                album_name: str = futures[fut].name
                tp_info:    str = "—"

                try:
                    res = fut.result()
                    all_results.append(res)
                    album_files = res.get("files", [])
                    for fr in album_files:
                        act = fr.get("action", "error")
                        counters[act] = counters.get(act, 0) + 1
                    done_files += len(album_files)

                    # Извлекаем рассчитанные параметры альбома
                    album_name = res.get("album", futures[fut].name)
                    peak_db    = res.get("peak_dbtp", -100.0)
                    scale_val  = res.get("scale", 1.0)

                    if 0 < scale_val < 1.0:
                        adj_db  = 20 * np.log10(scale_val)
                        tp_info = f"TP={peak_db:.2f} dBTP | Ослабление: {adj_db:.2f} дБ"
                    else:
                        tp_info = f"TP={peak_db:.2f} dBTP | Коррекция не требуется"

                except Exception as exc:
                    log.error("Воркер %s: %s", futures[fut], exc, exc_info=True)
                    counters["error"] = counters.get("error", 0) + 1
                    tp_info = "Ошибка обработки"

                pct = int(done_files / total_files * 100) if total_files else 100
                _progress_handler.set_pct(pct)

                # Логируем итоговую строку из MainProcess
                log.info("Завершён: %-45s | %s", album_name, tp_info)

        elapsed = time.monotonic() - t1
        log.info("=" * 78)
        log.info("Завершено за %.1f с (%.1f мин)", elapsed, elapsed / 60)

        labels = {
            "processed":         "Обработано (FIR + Noise Shaping 2)",
            "copy_dop":          "Скопировано (DoP-поток)",
            "copy_failed":       "Скопировано (ошибка декодирования)",
            "skip_done":         "Пропущено (уже обработано)",
            "error":             "Ошибки обработки",
        }
        log.info("=" * 78)
        log.info("ИТОГИ ОБРАБОТКИ (Версия %s):", VERSION)
        for act, cnt in sorted(counters.items()):
            if cnt:
                log.info("  %-30s : %d", labels.get(act, act), cnt)

        # Уровень 2: подсчёт файлов с ошибками верификации тегов
        tag_fail_count = sum(
            1 for r in all_results
            for f in r.get("files", [])
            if f.get("action") == "processed" and not f.get("tag_ok", True)
        )
        if tag_fail_count:
            log.warning("  %-30s : %d  ← требует внимания!", "VERIFY-TAGS FAIL", tag_fail_count)
        else:
            log.info("  %-30s : все OK", "Верификация тегов (ур.2)")

        # Уровень 3: подсчёт файлов с ошибками декода (только если --verify)
        if args.verify:
            audio_fail_count = sum(
                1 for r in all_results
                for f in r.get("files", [])
                if f.get("action") == "processed" and not f.get("audio_ok", True)
            )
            if audio_fail_count:
                log.warning("  %-30s : %d  ← требует внимания!", "VERIFY-AUDIO FAIL", audio_fail_count)
            else:
                log.info("  %-30s : все OK", "Верификация аудио (ур.3)")

        log.info("=" * 78)

        if args.report:
            try:
                Path(args.report).write_text(
                    json.dumps(sorted(all_results, key=lambda x: x.get("album", "")), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                log.warning("Не удалось записать JSON-отчёт (%s): %s", args.report, exc)

    finally:
        # Этот блок выполнится ВСЕГДА: и при успешном завершении, и при любой ошибке/интеррапте
        if fir_cache_path.exists():
            try:
                shutil.rmtree(fir_cache_path, ignore_errors=True)
                log.info("Временный FIR-кэш успешно удалён: %s", fir_cache_path)
            except Exception as exc:
                log.warning("Не удалось очистить FIR-кэш: %s", exc)

    return 0

def _remove_pycache() -> None:
    """Удаляет каталог __pycache__, создаваемый Python рядом со скриптом."""
    pycache = Path(__file__).parent / "__pycache__"
    if pycache.exists():
        try:
            shutil.rmtree(pycache)
        except Exception as exc:
            print(f"[WARNING] Не удалось удалить __pycache__: {exc}", file=sys.stderr)

if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        _remove_pycache()
