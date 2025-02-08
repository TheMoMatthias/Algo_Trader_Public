import requests
import os
import pandas as pd
from datetime import datetime, timedelta
from ratelimit import limits, sleep_and_retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from requests.exceptions import SSLError

# Basis-URL und Einstellungen für die API-Anfragen
spot_and_margin_url = "https://api.kucoin.com"
ANFRAGEN_LIMIT = 3000
ZEITRAUM = 30

@sleep_and_retry
@limits(calls=ANFRAGEN_LIMIT, period=ZEITRAUM)
def sichere_anfrage(url, max_retries=3):
    """Führt eine sichere Anfrage unter Berücksichtigung von Rate Limits und SSL-Fehlern durch."""
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 429:
                print("Rate Limit erreicht, warte für einige Sekunden...")
                time.sleep(10)
                continue
            elif response.status_code != 200:
                raise Exception(f'API response: {response.status_code}')
            return response.json()
        except SSLError as e:
            print(f"SSL-Fehler bei Versuch {i+1} für {url}: {e}")
            if i < max_retries - 1:
                time.sleep(5)
            else:
                raise

def get_service_status():
    """Ruft den aktuellen Service-Status der Kucoin API ab."""
    url = f"{spot_and_margin_url}/api/v1/status"
    return sichere_anfrage(url)["data"]

def get_symbols(endswith, margin=None):
    """
    Ruft eine Liste von Symbolen von Kucoin ab, die bestimmten Kriterien entsprechen.
    Parameters:
        endswith: Suffix der Symbolnamen, z.B. "-USDT".
        margin: None für beide, True für Margin enabled, False für Margin not enabled.
    Returns:
        Eine Liste gefilterter Symbole.
    """
    url = f"{spot_and_margin_url}/api/v2/symbols"
    data = sichere_anfrage(url)["data"]
    filtered_symbols = [
        i["symbol"] for i in data if i["symbol"].endswith(f"-{endswith}") and
        (margin is None or i.get("isMarginEnabled", False) == margin)
    ]
    return filtered_symbols

def get_symbol_start_date(symbol, earliest_date='2017-09-15'):
    """
    Ermittelt das früheste verfügbare Datum von Kursdaten für ein bestimmtes Symbol.
    Parameters:
        symbol: Das zu überprüfende Symbol.
        earliest_date: Das früheste Datum, ab dem gesucht werden soll.
    Returns:
        Das früheste verfügbare Datum als String.
    """
    earliest_timestamp = int(datetime.strptime(earliest_date, '%Y-%m-%d').timestamp())
    end_timestamp = int(datetime.now().timestamp())
    url = f"{spot_and_margin_url}/api/v1/market/candles?type=1day&symbol={symbol}&startAt={earliest_timestamp}&endAt={end_timestamp}"
    data = sichere_anfrage(url)["data"]
    if data:
        oldest_data = data[-1]
        oldest_timestamp = int(oldest_data[0])
        return datetime.fromtimestamp(oldest_timestamp).strftime('%Y-%m-%d')
    print(f"Keine historischen Daten gefunden für {symbol}.")
    return earliest_date

def letztes_update_finden(symbol, interval):
    """
    Findet das Datum des letzten Updates für ein bestimmtes Symbol und Intervall.
    Parameters:
        symbol: Das Symbol, für das das letzte Update gesucht wird.
        interval: Das Intervall der Daten.
    Returns:
        Das Datum des letzten Updates oder None, wenn keine Daten vorhanden sind.
    """
    directory = f"./data/{symbol}/{interval}"
    if not os.path.exists(directory) or not os.listdir(directory):
        return None
    latest_file = max(os.listdir(directory))
    return datetime.strptime(latest_file.split('_')[0], '%Y-%m-%d')

def get_kline_data(symbol, interval, update=False):
    """
    Ruft Kursdaten für ein bestimmtes Symbol und Intervall ab.
    Parameters:
        symbol: Das Symbol, für das Daten abgerufen werden sollen.
        interval: Das Intervall der Daten.
        update: Gibt an, ob eine Aktualisierung der Daten erforderlich ist.
    Returns:
        Eine Liste mit den abgerufenen Daten.
    """
    print(f"Herunterladen der Daten für {symbol}...")
    start_timestamp = int(datetime.strptime(get_symbol_start_date(symbol), '%Y-%m-%d').timestamp()) if not update else int(letztes_update_finden(symbol, interval).timestamp()) + 1
    end_timestamp = int((datetime.now() - timedelta(days=1)).timestamp())
    all_data = []
    while end_timestamp > start_timestamp:
        startAt = max(end_timestamp - 86400 * 30, start_timestamp)
        url = f"{spot_and_margin_url}/api/v1/market/candles?type={interval}&symbol={symbol}&startAt={startAt}&endAt={end_timestamp}"
        data = sichere_anfrage(url)["data"]
        if data:
            all_data.extend(data)
            end_timestamp = startAt - 1
        else:
            break
    return all_data


def save_data_to_excel(symbol, interval, data):
    """
    Aktualisiert die vorhandene Excel-Datei mit neuen Kursdaten oder erstellt eine neue, falls keine existiert.
    Parameters:
        symbol: Das Symbol, dessen Daten gespeichert werden sollen.
        interval: Das Intervall der Daten.
        data: Die zu speichernden Daten.
    """
    directory = f"./data/{symbol}/{interval}"
    os.makedirs(directory, exist_ok=True)

    # Finde die neueste vorhandene Excel-Datei
    files = sorted([f for f in os.listdir(directory) if f.endswith('.xlsx')], reverse=True)
    if files:
        latest_file = files[0]
        file_path = os.path.join(directory, latest_file)

        # Lese die vorhandene Excel-Datei ein
        existing_df = pd.read_excel(file_path)

        # Konvertiere Zeitstempel in das gleiche Format wie in den neuen Daten
        existing_df['time'] = pd.to_datetime(existing_df['time']).dt.strftime('%Y-%m-%d %H:%M')

        # Finde die Zeit des letzten Eintrags
        last_time = existing_df['time'].iloc[-1]

        # Filtere alle neuen Daten, die nach dem letzten Zeitpunkt der vorhandenen Daten liegen
        new_data_df = pd.DataFrame(data, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        new_data_df['time'] = pd.to_datetime(new_data_df['time'].astype(int), unit='s').dt.strftime('%Y-%m-%d %H:%M')
        new_data_to_append = new_data_df[new_data_df['time'] > last_time]

        # Hänge die neuen Daten an die vorhandene Datei an
        updated_df = existing_df.append(new_data_to_append, ignore_index=True)
        updated_df.to_excel(file_path, index=False)
        print(f"Die vorhandene Datei {latest_file} wurde mit neuen Daten aktualisiert.")
    else:
        # Wenn keine Datei existiert, erstelle eine neue
        new_data_df = pd.DataFrame(data, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'turnover'])
        new_data_df['time'] = pd.to_datetime(new_data_df['time'].astype(int), unit='s').dt.strftime('%Y-%m-%d %H:%M')
        filename = f"{directory}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
        new_data_df.to_excel(filename, index=False)
        print(f"Neue Daten für {symbol} gespeichert in {filename}")


def fetch_and_save(symbol, intervals):
    """
    Verarbeitet das Herunterladen und Speichern von Kursdaten für gegebene Symbole und Intervalle.
    Parameters:
        symbol: Das Symbol, für das Daten verarbeitet werden sollen.
        intervals: Eine Liste der Intervalle, für die Daten verarbeitet werden sollen.
    """
    for interval in intervals:
        update = os.path.exists(f"./data/{symbol}/{interval}")
        data = get_kline_data(symbol, interval, update=update)
        save_data_to_excel(symbol, interval, data)

def main(symbols, intervals, max_workers=10):
    """
    Hauptfunktion zur Steuerung des parallelen Herunterladens und Speicherns von Kursdaten.
    Parameters:
        symbols: Eine Liste der Symbole, für die Daten heruntergeladen werden sollen.
        intervals: Eine Liste der Intervalle, für die Daten heruntergeladen werden sollen.
        max_workers: Die maximale Anzahl von Threads für den parallelen Download.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_and_save, symbol, intervals) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()

def print_directory_size(directory):
    """Druckt die Gesamtgröße des angegebenen Verzeichnisses."""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    print(f"Gesamtgröße des Ordners '{directory}': {total_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    symbols = get_symbols("USDT")
    intervals = ["5min", "1hour", "1day"]
    main(symbols, intervals)
    print_directory_size("./data")
