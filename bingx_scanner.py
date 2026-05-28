import os
import time
import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import requests

# Импорт клиента BingxClient из предоставленного модуля
from bingx_client import BingxClient

# ==============================================================================
# --- НАСТРОЙКИ СКАНИРОВАНИЯ ---
# ==============================================================================
SCAN_INTERVAL_SECONDS: int = 60      # Интервал сканирования в секундах
TIMEFRAME: str = "1h"                # Таймфрейм для свечей индикаторов (15m, 1h, 4h)
KLINES_LIMIT: int = 1000              # Количество свечей для расчёта EMA и MACD

# --- ПОРОГИ ОСНОВНЫХ ФИЛЬТРОВ ---
OI_THRESHOLD: float = 1_000_000.0    # Минимальный Open Interest в USDT
OI_CHANGE_PERCENT: float = 1.0      # Минимальный рост OI в процентах (если используется)
FUNDING_RATE_THRESHOLD: float = 0.0005 # 0.5% (абсолютное значение, напр. 0.005 = 0.5%)

# --- НАСТРОЙКИ ТЕХНИЧЕСКИХ ИНДИКАТОРОВ ---
EMA_FAST: int = 9
EMA_SLOW: int = 21

MACD_FAST: int = 12
MACD_SLOW: int = 26
MACD_SIGNAL: int = 9

# --- РЕЖИМЫ РАБОТЫ ---
USE_OI_CHANGE: bool = True          # True = процентное изменение OI, False = абсолютный порог OI
OI_CHANGE_WINDOW: int = 24           # Окно свечей для изменения OI (если USE_OI_CHANGE = True)
FILTER_MODE: str = "AND"             # "AND" = оба фильтра выполняются, "OR" = хотя бы один

# --- НАСТРОЙКИ ТЕЛЕГРАМ ---
TELEGRAM_COOLDOWN_HOURS: float = 4.0 # Кулдаун повторной отправки сигнала (в часах)
# ==============================================================================


def get_timeframe_minutes(timeframe: str) -> int:
    """
    Преобразует строку таймфрейма (напр. '15m', '1h', '1d') в количество минут.
    
    Args:
        timeframe (str): Таймфрейм в виде строки.
        
    Returns:
        int: Количество минут.
    """
    try:
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 60 * 24
    except Exception:
        pass
    return 60  # Значение по умолчанию


class OIHistoryTracker:
    """
    Класс для накопления и отслеживания истории Open Interest (открытого интереса)
    в оперативной памяти с целью расчёта процентного изменения за период.
    """
    def __init__(self, window_candles: int, timeframe: str):
        self.window_candles: int = window_candles
        self.timeframe: str = timeframe
        # Структура: { symbol: [(timestamp, oi_value), ...] }
        self.history: Dict[str, List[Tuple[float, float]]] = {}
        
    def add_entry(self, symbol: str, oi: float) -> None:
        """Добавляет новое измерение OI в историю и очищает старые записи."""
        if symbol not in self.history:
            self.history[symbol] = []
        now = time.time()
        self.history[symbol].append((now, oi))
        
        # Рассчитываем максимальное время хранения данных:
        # window_candles * минут в свече * 60 сек, добавляем 50% буфера
        timeframe_mins = get_timeframe_minutes(self.timeframe)
        max_age_seconds = self.window_candles * timeframe_mins * 60
        cutoff = now - max_age_seconds * 1.5
        
        # Очищаем историю от слишком старых записей для экономии памяти
        self.history[symbol] = [entry for entry in self.history[symbol] if entry[0] > cutoff]
        
    def get_percentage_change(self, symbol: str, current_oi: float) -> Tuple[Optional[float], str]:
        """
        Рассчитывает процент изменения OI относительно значения N свечей назад.
        
        Returns:
            Tuple[Optional[float], str]: Процент изменения (или None) и статус-сообщение.
        """
        if symbol not in self.history or len(self.history[symbol]) < 2:
            return None, "Накапливается история (нужно минимум 2 сканирования)"
            
        timeframe_mins = get_timeframe_minutes(self.timeframe)
        target_age_seconds = self.window_candles * timeframe_mins * 60
        now = time.time()
        target_time = now - target_age_seconds
        
        history = self.history[symbol]
        # Находим запись, наиболее близкую к целевому времени
        closest_entry = min(history, key=lambda x: abs(x[0] - target_time))
        
        # Если ближайшая запись слишком свежая (меньше 80% от целевого времени),
        # мы возвращаем сравнение с самой старой имеющейся записью, но помечаем статус.
        if (now - closest_entry[0]) < target_age_seconds * 0.8:
            oldest_entry = history[0]
            actual_age_mins = (now - oldest_entry[0]) / 60.0
            historical_oi = oldest_entry[1]
            if historical_oi == 0:
                return 0.0, f"Накапливается история ({actual_age_mins:.1f} мин.)"
            change = ((current_oi - historical_oi) / historical_oi) * 100.0
            return change, f"Накапливается история ({actual_age_mins:.1f} мин. из {target_age_seconds/60:.1f} мин.)"
        else:
            historical_oi = closest_entry[1]
            if historical_oi == 0:
                return 0.0, "Базовое значение OI = 0"
            change = ((current_oi - historical_oi) / historical_oi) * 100.0
            return change, "История полностью накоплена"


def calculate_indicators(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Расчёт экспоненциальных скользящих средних (EMA) и гистограммы MACD с использованием pandas.
    
    Args:
        df (pd.DataFrame): DataFrame со свечными данными (должна содержать колонку 'close').
        
    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (ema_fast, ema_slow, macd_hist)
    """
    # Расчёт EMA
    ema_fast = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    
    # Расчёт MACD
    macd_line = df['close'].ewm(span=MACD_FAST, adjust=False).mean() - df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    macd_signal = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    
    return ema_fast, ema_slow, macd_hist


def check_technical_signals(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Анализирует пересечения EMA и гистограммы MACD на завершённых свечах.
    
    Для исключения шумов вычисления проводятся на закрытых свечах:
      - Индекс -2 (последняя закрытая свеча)
      - Индекс -3 (предыдущая закрытая свеча)
      - Индекс -4 (свеча перед ней, для поиска пересечений с лагом в 1 свечу)
      
    Args:
        df (pd.DataFrame): Свечные данные от API.
        
    Returns:
        Dict[str, Any]: Результаты анализа (signal, ema_fast_val, ema_slow_val, macd_hist_val, details).
    """
    result = {
        "signal": "NEUTRAL",
        "ema_fast_val": 0.0,
        "ema_slow_val": 0.0,
        "macd_hist_val": 0.0,
        "details": ""
    }
    
    min_required_len = max(EMA_SLOW, MACD_SLOW + MACD_SIGNAL) + 5
    if len(df) < min_required_len:
        result["details"] = f"Недостаточно свечей ({len(df)} < {min_required_len})"
        return result
        
    ema_fast, ema_slow, macd_hist = calculate_indicators(df)
    
    # Текущие закрытые значения (индекс -2)
    val_fast_curr = float(ema_fast.iloc[-2])
    val_slow_curr = float(ema_slow.iloc[-2])
    hist_curr = float(macd_hist.iloc[-2])
    
    # Предыдущие закрытые значения (индекс -3)
    val_fast_prev = float(ema_fast.iloc[-3])
    val_slow_prev = float(ema_slow.iloc[-3])
    hist_prev = float(macd_hist.iloc[-3])
    
    # Значения позапрошлые (индекс -4) для поиска пересечений с небольшим лагом
    val_fast_prev2 = float(ema_fast.iloc[-4])
    val_slow_prev2 = float(ema_slow.iloc[-4])
    hist_prev2 = float(macd_hist.iloc[-4])
    
    result["ema_fast_val"] = val_fast_curr
    result["ema_slow_val"] = val_slow_curr
    result["macd_hist_val"] = hist_curr
    
    # Бычий/медвежий статус на последней закрытой свече
    ema_bullish = val_fast_curr > val_slow_curr
    ema_bearish = val_fast_curr < val_slow_curr
    macd_bullish = hist_curr > 0
    macd_bearish = hist_curr < 0
    
    # 1. Пересечение EMA
    ema_cross_up = (val_fast_prev <= val_slow_prev) and (val_fast_curr > val_slow_curr)
    ema_cross_down = (val_fast_prev >= val_slow_prev) and (val_fast_curr < val_slow_curr)
    # С лагом в 1 свечу
    ema_cross_up_lag = (val_fast_prev2 <= val_slow_prev2) and (val_fast_prev > val_slow_prev)
    ema_cross_down_lag = (val_fast_prev2 >= val_slow_prev2) and (val_fast_prev < val_slow_prev)
    
    # 2. Пересечение MACD через ноль
    macd_cross_up = (hist_prev <= 0.0) and (hist_curr > 0.0)
    macd_cross_down = (hist_prev >= 0.0) and (hist_curr < 0.0)
    # С лагом в 1 свечу
    macd_cross_up_lag = (hist_prev2 <= 0.0) and (hist_prev > 0.0)
    macd_cross_down_lag = (hist_prev2 >= 0.0) and (hist_prev < 0.0)
    
    # Сигнал формируется, если ОБА индикатора согласованы (бычьи / медвежьи)
    # И хотя бы у одного произошло пересечение в пределах последних двух закрытых свечей (текущая закрытая или предыдущая).
    is_buy = (ema_bullish and macd_bullish) and (ema_cross_up or macd_cross_up or ema_cross_up_lag or macd_cross_up_lag)
    is_sell = (ema_bearish and macd_bearish) and (ema_cross_down or macd_cross_down or ema_cross_down_lag or macd_cross_down_lag)
    
    if is_buy:
        result["signal"] = "BUY"
        cross_desc = []
        if ema_cross_up or ema_cross_up_lag:
            cross_desc.append("пересечение EMA вверх")
        if macd_cross_up or macd_cross_up_lag:
            cross_desc.append("пересечение MACD вверх")
        result["details"] = " и ".join(cross_desc) if cross_desc else "тренд подтвержден"
    elif is_sell:
        result["signal"] = "SELL"
        cross_desc = []
        if ema_cross_down or ema_cross_down_lag:
            cross_desc.append("пересечение EMA вниз")
        if macd_cross_down or macd_cross_down_lag:
            cross_desc.append("пересечение MACD вниз")
        result["details"] = " и ".join(cross_desc) if cross_desc else "тренд подтвержден"
    else:
        result["signal"] = "NEUTRAL"
        result["details"] = f"Нет согласованного сигнала (EMA: {'Бычий' if ema_bullish else 'Медвежий'}, MACD: {'Бычий' if macd_bullish else 'Медвежий'})"
        
    return result


def generate_chart(symbol: str, df_klines: pd.DataFrame) -> Optional[str]:
    """
    Генерирует свечной график с EMA и гистограммой MACD, сохраняет во временный файл.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        # Считаем индикаторы для всего датафрейма (чтобы избежать краевых эффектов)
        ema_fast, ema_slow, macd_hist = calculate_indicators(df_klines)
        
        # Для визуализации берем последние 45 свечей
        tail_size = 45
        df = df_klines.tail(tail_size).copy()
        ema_f = ema_fast.tail(tail_size)
        ema_s = ema_slow.tail(tail_size)
        m_hist = macd_hist.tail(tail_size)
        
        # Создаем фигуру с двумя сабплотами (верхний для цены, нижний для MACD)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
        fig.patch.set_facecolor('#121212') # Sleek dark mode background
        
        ax1.set_facecolor('#1a1a1a')
        ax2.set_facecolor('#1a1a1a')
        
        # Индексы для оси X
        x = np.arange(len(df))
        
        # Отрисовка свечей
        for i in range(len(df)):
            row = df.iloc[i]
            # Свеча растет (зеленая) или падает (красная)
            color = '#00c087' if row['close'] >= row['open'] else '#ff3b30'
            
            # Фитиль (High-Low)
            ax1.vlines(x[i], row['low'], row['high'], color=color, linewidth=1.5)
            # Тело (Open-Close)
            body_bottom = min(row['open'], row['close'])
            body_height = abs(row['close'] - row['open'])
            if body_height == 0:
                body_height = (row['high'] - row['low']) * 0.05
            
            ax1.bar(x[i], body_height, bottom=body_bottom, color=color, width=0.6, align='center', edgecolor=color, linewidth=0.5)
            
        # Рисуем линии EMA
        ax1.plot(x, ema_f, color='#2f80ed', linewidth=1.5, label=f'EMA({EMA_FAST})')
        ax1.plot(x, ema_s, color='#f2994a', linewidth=1.5, label=f'EMA({EMA_SLOW})')
        
        ax1.set_title(f"{symbol} ({TIMEFRAME}) - Сигнальный график", color='white', fontsize=14, fontweight='bold', pad=10)
        ax1.grid(True, color='#2c2c2c', linestyle='--', alpha=0.5)
        ax1.tick_params(colors='white')
        ax1.legend(loc='upper left', facecolor='#1a1a1a', edgecolor='#2c2c2c', labelcolor='white')
        
        # Отрисовка MACD гистограммы
        for i in range(len(m_hist)):
            val = m_hist.iloc[i]
            color = '#00c087' if val >= 0 else '#ff3b30'
            ax2.bar(x[i], val, color=color, width=0.6, align='center')
            
        ax2.set_title("MACD гистограмма", color='white', fontsize=10, pad=5)
        ax2.grid(True, color='#2c2c2c', linestyle='--', alpha=0.5)
        ax2.tick_params(colors='white')
        
        # Настройка временных меток на оси X
        step = max(1, len(df) // 8)
        xticks = x[::step]
        xticklabels = [df['time'].iloc[i].strftime('%d.%m %H:%M') for i in xticks]
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xticklabels, rotation=30, ha='right', color='white')
        
        plt.tight_layout()
        
        # Папка для сохранения временных файлов
        os.makedirs("scratch", exist_ok=True)
        chart_path = f"scratch/{symbol.replace('/', '_')}_chart.png"
        plt.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=120)
        plt.close()
        
        return chart_path
    except Exception as e:
        print(f"[CHART ERROR] Ошибка генерации свечного графика для {symbol}: {e}")
        return None


def send_telegram_signal(symbol: str, direction: str, mark_price: float, oi: float, oi_status: str,
                         funding_rate: float, ema_fast_val: float, ema_slow_val: float, macd_val: float,
                         analysis_details: str, df_klines: pd.DataFrame) -> None:
    """
    Формирует красивое HTML-сообщение в Telegram, генерирует свечной график
    и отправляет фото с описанием в Telegram.
    """
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("[TG] Переменные TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID не заданы. Пропуск отправки в Telegram.")
        return

    # 1. Генерация графика
    chart_path = generate_chart(symbol, df_klines)
    
    # 2. Формирование сообщения
    emoji = "🟢" if direction == "BUY" else "🔴"
    dir_ru = "ЛОНГ (BUY)" if direction == "BUY" else "ШОРТ (SELL)"
    funding_pct = funding_rate * 100
    
    # Красивая верстка в HTML
    text = (
        f"{emoji} <b>СИГНАЛ: {dir_ru} для {symbol}</b>\n\n"
        f"💵 <b>Цена маркировки:</b> <code>{mark_price:.5f}</code>\n"
        f"📊 <b>Open Interest:</b> <code>{oi:,.2f} USDT</code> ({oi_status})\n"
        f"💸 <b>Funding Rate:</b> <code>{funding_pct:+.4f}%</code> (8h)\n\n"
        f"📈 <b>EMA({EMA_FAST}) &gt; EMA({EMA_SLOW}):</b> <code>{ema_fast_val:.5f}</code> {'&gt;' if ema_fast_val > ema_slow_val else '&lt;'} <code>{ema_slow_val:.5f}</code>\n"
        f"📉 <b>MACD гистограмма:</b> <code>{macd_val:.5f}</code> ({analysis_details})\n\n"
        f"🕒 <i>Время сигнала: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>"
    )
    
    # 3. Отправка фото в Telegram
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    
    try:
        if chart_path and os.path.exists(chart_path):
            with open(chart_path, 'rb') as photo:
                payload = {
                    "chat_id": chat_id,
                    "caption": text,
                    "parse_mode": "HTML"
                }
                files = {"photo": photo}
                r = requests.post(url, data=payload, files=files, timeout=15)
            # Удаляем временный файл графика
            try:
                os.remove(chart_path)
            except Exception:
                pass
        else:
            # Если график не сгенерировался, шлем обычным текстом
            url_msg = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            r = requests.post(url_msg, data=payload, timeout=15)
            
        if r.status_code == 200:
            print(f"[TG] Сигнал {direction} по {symbol} успешно отправлен в Telegram!")
        else:
            print(f"[TG] Ошибка отправки в Telegram: Код {r.status_code}, Ответ: {r.text}")
    except Exception as e:
        print(f"[TG] Исключение при отправке в Telegram: {e}")


def main(client: BingxClient) -> None:
    """
    Основной бесконечный цикл сканера фьючерсного рынка BingX.
    
    Шаги работы:
      1. Запрашивает список всех активных контрактов.
      2. Делает групповой (bulk) запрос цен и ставок финансирования всех токенов.
      3. Применяет фильтры Open Interest и Funding Rate.
      4. По прошедшим фильтр токенам запрашивает историю свечей.
      5. Проводит технический анализ (EMA, MACD).
      6. При обнаружении сигналов выводит структурированное уведомление в консоль.
    """
    print("======================================================================")
    print("🚀 ЗАПУСК ТОРГОВОЙ СИСТЕМЫ СКАНИРОВАНИЯ BINGX")
    print(f"🕒 Время старта: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("----------------------------------------------------------------------")
    print(f"📊 Настройки индикаторов: Таймфрейм = {TIMEFRAME}, Свечей для расчёта = {KLINES_LIMIT}")
    print(f"📈 Параметры EMA: Fast = {EMA_FAST}, Slow = {EMA_SLOW}")
    print(f"📉 Параметры MACD: Fast = {MACD_FAST}, Slow = {MACD_SLOW}, Signal = {MACD_SIGNAL}")
    print(f"⚙️ Режим фильтрации: {FILTER_MODE}")
    print(f"💰 Порог ставки финансирования: {FUNDING_RATE_THRESHOLD * 100:.4f}%")
    if USE_OI_CHANGE:
        print(f"⏳ Порог изменения Open Interest: >= {OI_CHANGE_PERCENT}% (за {OI_CHANGE_WINDOW} свечей)")
    else:
        print(f"💎 Порог Open Interest: >= {OI_THRESHOLD:,} USDT (абсолютное значение)")
    print("======================================================================\n")
    
    oi_tracker = OIHistoryTracker(window_candles=OI_CHANGE_WINDOW, timeframe=TIMEFRAME)
    signal_cooldowns: Dict[Tuple[str, str], float] = {}
    
    while True:
        try:
            start_time = time.time()
            scan_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{scan_time_str}] Запуск нового цикла сканирования...")
            
            # 1. Получаем список активных фьючерсных токенов
            tickers = client.get_all_tikers()
            if not tickers:
                print(f"[{scan_time_str}] [WARNING] Не удалось получить список торговых пар от BingX. Повтор через {SCAN_INTERVAL_SECONDS}с.")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue
            
            ticker_set = set(tickers)
            
            # 2. Оптимизированный групповой (bulk) запрос цен и финансирования
            premium_indices = client.get_premium_index()
            if not premium_indices:
                print(f"[{scan_time_str}] [WARNING] Не удалось получить премиум-индексы. Повтор через {SCAN_INTERVAL_SECONDS}с.")
                time.sleep(SCAN_INTERVAL_SECONDS)
                continue
                
            # Группируем полученные данные по символам
            market_data: Dict[str, Dict[str, float]] = {}
            for item in premium_indices:
                symbol = item.get("symbol")
                if symbol in ticker_set:
                    try:
                        market_data[symbol] = {
                            "funding_rate": float(item.get("lastFundingRate") or 0.0),
                            "mark_price": float(item.get("markPrice") or 0.0),
                            "index_price": float(item.get("indexPrice") or 0.0)
                        }
                    except (ValueError, TypeError):
                        continue
                        
            print(f"[{scan_time_str}] Получены рыночные данные для {len(market_data)} фьючерсных пар.")
            
            # 3. Фильтрация по Funding Rate
            passed_funding: List[str] = []
            for symbol, data in market_data.items():
                if abs(data["funding_rate"]) >= FUNDING_RATE_THRESHOLD:
                    passed_funding.append(symbol)
                    
            print(f"[{scan_time_str}] Фильтр ставки финансирования (>= {FUNDING_RATE_THRESHOLD*100:.4f}%) прошли {len(passed_funding)} пар.")
            
            # 4. Проверка фильтра Open Interest и составление списка кандидатов
            candidates: List[Tuple[str, float, str]] = []
            
            if FILTER_MODE == "AND":
                # В режиме AND запрашиваем OI только для тех, кто прошёл Funding Rate фильтр
                # Это минимизирует количество запросов и исключает rate limit
                for symbol in passed_funding:
                    try:
                        time.sleep(0.2)  # Безопасная задержка для API
                        oi = client.get_open_insterest(symbol)
                        if oi is None:
                            continue
                            
                        oi_tracker.add_entry(symbol, oi)
                        
                        oi_passed = False
                        oi_status_msg = ""
                        
                        if USE_OI_CHANGE:
                            oi_change, oi_status_msg = oi_tracker.get_percentage_change(symbol, oi)
                            if oi_change is not None:
                                if abs(oi_change) >= OI_CHANGE_PERCENT:
                                    oi_passed = True
                                    oi_status_msg = f"{oi_change:+.2f}% ({oi_status_msg})"
                            else:
                                # Резервный абсолютный фильтр при первом запуске (пока история накапливается)
                                if oi >= OI_THRESHOLD:
                                    oi_passed = True
                                    oi_status_msg = f"Первый запуск (абсолютный OI {oi:,.0f} >= {OI_THRESHOLD:,.0f})"
                                else:
                                    oi_status_msg = f"Первый запуск ({oi_status_msg})"
                        else:
                            if oi >= OI_THRESHOLD:
                                oi_passed = True
                                oi_status_msg = "Абсолютный порог"
                                
                        if oi_passed:
                            candidates.append((symbol, oi, oi_status_msg))
                    except Exception as e:
                        print(f"[{scan_time_str}] [ERROR] Ошибка при проверке OI для {symbol}: {e}")
                        continue
            
            else:  # FILTER_MODE == "OR"
                # В режиме OR нам нужно проверять все пары на OI, что может вызвать задержки
                print(f"[{scan_time_str}] [WARNING] Внимание: режим OR опрашивает все {len(market_data)} пар. Это может занять до 2-3 минут.")
                for symbol in market_data.keys():
                    try:
                        # Если пара уже прошла Funding Rate, она добавляется в кандидаты автоматически
                        if symbol in passed_funding:
                            time.sleep(0.1)
                            oi = client.get_open_insterest(symbol) or 0.0
                            oi_tracker.add_entry(symbol, oi)
                            candidates.append((symbol, oi, "Прошёл по Funding Rate"))
                            continue
                            
                        # Иначе опрашиваем и проверяем OI
                        time.sleep(0.1)
                        oi = client.get_open_insterest(symbol)
                        if oi is None:
                            continue
                            
                        oi_tracker.add_entry(symbol, oi)
                        
                        oi_passed = False
                        oi_status_msg = ""
                        
                        if USE_OI_CHANGE:
                            oi_change, oi_status_msg = oi_tracker.get_percentage_change(symbol, oi)
                            if oi_change is not None:
                                if abs(oi_change) >= OI_CHANGE_PERCENT:
                                    oi_passed = True
                                    oi_status_msg = f"{oi_change:+.2f}% ({oi_status_msg})"
                            else:
                                # Резервный абсолютный фильтр при первом запуске (пока история накапливается)
                                if oi >= OI_THRESHOLD:
                                    oi_passed = True
                                    oi_status_msg = f"Первый запуск (абсолютный OI {oi:,.0f} >= {OI_THRESHOLD:,.0f})"
                                else:
                                    oi_status_msg = f"Первый запуск ({oi_status_msg})"
                        else:
                            if oi >= OI_THRESHOLD:
                                oi_passed = True
                                oi_status_msg = "Абсолютный порог"
                                
                        if oi_passed:
                            candidates.append((symbol, oi, oi_status_msg))
                    except Exception as e:
                        continue
                        
            print(f"[{scan_time_str}] Первичную фильтрацию ({FILTER_MODE}) успешно прошли {len(candidates)} пар.")
            
            # 5. Детальный технический анализ кандидатов
            for symbol, oi, oi_status in candidates:
                try:
                    time.sleep(0.2)  # Безопасный интервал между запросами свечей
                    
                    df_klines = client.get_klines(symbol, TIMEFRAME, limit=KLINES_LIMIT)
                    if df_klines.empty:
                        continue
                        
                    analysis = check_technical_signals(df_klines)
                    
                    if analysis["signal"] in ("BUY", "SELL"):
                        mark_price = market_data[symbol]["mark_price"]
                        funding_rate = market_data[symbol]["funding_rate"]
                        
                        # Вывод оформленного сигнала в консоль
                        direction_str = "BUY (LONG)" if analysis["signal"] == "BUY" else "SELL (SHORT)"
                        funding_pct_str = f"{funding_rate * 100:+.4f}%"
                        
                        ema_fast_val = analysis["ema_fast_val"]
                        ema_slow_val = analysis["ema_slow_val"]
                        macd_val = analysis["macd_hist_val"]
                        
                        print("\n==================================================")
                        print(f"[{scan_time_str}] СИГНАЛ: {direction_str} для {symbol}")
                        print(f"   Цена маркировки: {mark_price:.5f}")
                        print(f"   Open Interest: {oi:,.2f} USDT ({oi_status})")
                        print(f"   Funding Rate: {funding_pct_str} (8h)")
                        print(f"   EMA({EMA_FAST}) > EMA({EMA_SLOW}): {ema_fast_val:.5f} {'/ > /' if ema_fast_val > ema_slow_val else '/ < /'} {ema_slow_val:.5f}")
                        print(f"   MACD гистограмма: {macd_val:.5f} ({analysis['details']})")
                        print("==================================================")
                        
                        # --- Отправка в Telegram с контролем кулдауна ---
                        sig_key = (symbol, analysis["signal"])
                        now = time.time()
                        last_sent = signal_cooldowns.get(sig_key, 0.0)
                        if (now - last_sent) >= (TELEGRAM_COOLDOWN_HOURS * 3600):
                            signal_cooldowns[sig_key] = now
                            send_telegram_signal(
                                symbol=symbol,
                                direction=analysis["signal"],
                                mark_price=mark_price,
                                oi=oi,
                                oi_status=oi_status,
                                funding_rate=funding_rate,
                                ema_fast_val=ema_fast_val,
                                ema_slow_val=ema_slow_val,
                                macd_val=macd_val,
                                analysis_details=analysis["details"],
                                df_klines=df_klines
                            )
                        else:
                            cooldown_left_sec = (TELEGRAM_COOLDOWN_HOURS * 3600) - (now - last_sent)
                            print(f"[TG INFO] Сигнал {analysis['signal']} по {symbol} пропущен (кулдаун, осталось {cooldown_left_sec / 3600:.2f} ч.)")
                        
                except Exception as ex:
                    print(f"[{scan_time_str}] [ERROR] Не удалось рассчитать индикаторы для {symbol}: {ex}")
                    continue
            
            # Расчёт времени сна
            elapsed = time.time() - start_time
            sleep_time = max(1.0, SCAN_INTERVAL_SECONDS - elapsed)
            print(f"[{scan_time_str}] Сканирование завершено за {elapsed:.2f} сек. Сон {sleep_time:.2f} сек...\n")
            time.sleep(sleep_time)
            
        except Exception as e:
            print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [CRITICAL ERROR] Исключение в основном потоке: {e}")
            time.sleep(10)


if __name__ == "__main__":
    api_key = ''
    api_secret = ''
    
    # Создаем экземпляр BingxClient (для публичных запросов API ключи могут быть пустыми)
    client = BingxClient(api_key, api_secret, testnet=True)
  
    try:
        main(client)
    except KeyboardInterrupt:
        print("\nСканер остановлен пользователем. До свидания!")
