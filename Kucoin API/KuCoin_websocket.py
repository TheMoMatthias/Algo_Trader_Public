import configparser
import time
import requests
import datetime
import base64
import hmac
import hashlib
import json
import websocket
import threading
from datetime import datetime, timezone
import logging
import os


class ColorFormatter(logging.Formatter):
    '''
    class for logging, includes colour's for better log vizualisation and two functions for converting datetime in ms format
    '''
    format = "%(asctime)s [%(levelname)s] %(message)s"
    FORMATS = {
        logging.DEBUG: "\033[94m" + format + "\033[0m",  # Blue
        logging.INFO: "\033[92m" + format + "\033[0m",  # Green
        logging.WARNING: "\033[93m" + format + "\033[0m",  # Gelb
        logging.ERROR: "\033[91m" + format + "\033[0m",  # Red
        logging.CRITICAL: "\033[91m\033[1m" + format + "\033[0m",  # Red + Bold
    }

    def formatTime(self, record, datefmt=None):
        # Erzeugt ein datetime-Objekt aus dem timestamp
        record_datetime = datetime.fromtimestamp(record.created)
        # Formatierung des datetime-Objekts mit Millisekunden
        if datefmt:
            formatted_time = record_datetime.strftime(datefmt)
        else:
            formatted_time = record_datetime.strftime("%Y-%m-%d %H:%M:%S")
        # Füge die Millisekunden hinzu
        formatted_time_with_ms = f"{formatted_time}.{int(record.msecs):03d}"
        return formatted_time_with_ms

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self.format)
        self._style._fmt = log_fmt
        # Nutze die Basis-Klasse, um das Format festzulegen
        formatter = logging.Formatter(self._style._fmt, datefmt='%Y-%m-%d %H:%M:%S')
        # Setze das korrekte Datum und die Uhrzeit mit Millisekunden
        record.asctime = self.formatTime(record)
        return super().format(record)


# # Logger-Configuration
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)

# # File Handler
# base_path = os.path.dirname(os.path.realpath(__file__))
# file_handler = logging.FileHandler(f'{base_path}/websocket_log.txt', mode='a')
# file_format = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S.')
# file_handler.setFormatter(file_format)
# file_handler.setLevel(logging.DEBUG)
# logger.addHandler(file_handler)

# Terminal-Handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(ColorFormatter())
stream_handler.setLevel(logging.DEBUG)
# logger.addHandler(stream_handler)


#################################################################################################################################################################
#
#
#                                                                  KUCOIN WebSocket API CLASS
#
#################################################################################################################################################################


class KucoinWebSocketClient:
    def __init__(self, config_file='JansConfig.ini'):
        """
        initilize the client
        """
        self.config = configparser.ConfigParser()
        base_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(base_dir,"..", "Config", config_file)
        self.config.read(config_path)
        self.api_key = self.config["kucoin_api_keys"]["api_key"]
        self.api_secret = self.config["kucoin_api_keys"]["api_secret"]
        self.api_passphrase = self.config["kucoin_api_keys"]["api_passphrase"]
        self.spot_and_margin_url = "https://api.kucoin.com"
        self.ws = None
        self.token = None
        self.instance_servers = None 
        self.ping_interval = None  # recommended Interval for sending 'ping' to server to maintain the connection 
        self.ping_timeout = None  # After such a long time(seconds), if you do not receive pong, it will be considered as disconnected.
        self.last_ping_time = 0
        self.shutting_down = False
        self.reconnecting = False
        self.latest_price_data = {}
        self.data_lock = threading.Lock()
        self.topics = {
            "ticker": "/market/ticker:{}",  # get the specified [symbol] push of BBO changes; Push frequency: once every 100ms
            "index": "/indicator/index:{}",  # get the mark price for margin trading.
            "markPrice": "/indicator/markPrice:{}",  # get the index price for the margin trading.
            # "positions": "/margin/position"
            # add here endpoints of websockt URL's if needed   
        }     
    
    def get_server_timestamp(self):
        response = requests.get(f"{self.spot_and_margin_url}/api/v1/timestamp")
        if response.status_code == 200:
            timestamp = response.json()["data"] / 1000
            # logger.info("Fetching server timestamp...")
            return datetime.utcfromtimestamp(timestamp)
        else:
            raise Exception(f"Failed to fetch server time, status code: {response.status_code}")

    def request_token(self, private=False):
        url = f"{self.spot_and_margin_url}/api/v1/bullet-private" if private else f"{self.spot_and_margin_url}/api/v1/bullet-public"
        if private:
            now = int(time.time() * 1000)
            str_to_sign = str(now) + 'POST' + '/api/v1/bullet-private'
            signature = base64.b64encode(hmac.new(self.api_secret.encode('utf-8'), str_to_sign.encode('utf-8'), hashlib.sha256).digest())
            api_passphrase_encoded = base64.b64encode(hmac.new(self.api_secret.encode('utf-8'), self.api_passphrase.encode('utf-8'), hashlib.sha256).digest())
            headers = {
                "KC-API-SIGN": signature,
                "KC-API-TIMESTAMP": str(now),
                "KC-API-KEY": self.api_key,
                "KC-API-PASSPHRASE": api_passphrase_encoded,
                "KC-API-KEY-VERSION": "2"
            }
            response = requests.post(url, headers=headers)
        else:
            response = requests.post(url)

        if response.status_code == 200:
            self.token = response.json()['data']['token']
            self.instance_servers = response.json()['data']['instanceServers'][0]
            self.ping_interval = self.instance_servers['pingInterval'] / 1000  # Convert milliseconds to seconds
            self.ping_timeout = self.instance_servers['pingTimeout'] / 1000  # Convert milliseconds to seconds
            self.init_websocket(self.instance_servers['endpoint'])
        else:
            raise Exception(f"Token request failed: {response.status_code}")

    def init_websocket(self, endpoint):
        self.ws = websocket.WebSocketApp(f"{endpoint}?token={self.token}",
                                         on_open=self.on_open,
                                         on_message=self.on_message,
                                         on_error=self.on_error,
                                         on_close=self.on_close)
        self.last_ping_time = time.time()
        websocket_thread = threading.Thread(target=self.run_forever)
        websocket_thread.start()
        if time.time() - self.last_ping_time >= self.ping_interval:
                self.send_ping()
    
    def run_forever(self):
        self.ws.run_forever()
    
    def on_open(self, ws):
        self.last_ping_time = time.time()
        threading.Thread(target=self.ping_thread).start()

    def on_message(self, ws, message):
        print(message)
        # logger.debug(f"{datetime.now()}: {message}")
        message_data = json.loads(message)
        
        if message_data.get('type') == "pong":
            self.last_pong_time = time.time() * 1000

        elif message_data.get('type') == 'message' and message_data.get('topic').startswith("/market/ticker:"):
            self.ws_priceData(message_data)
       
        current_time = time.time()
        if current_time - self.last_ping_time >= self.ping_interval:
            self.send_ping(ws)

    def on_error(self, ws, error):  
        # logger.error(f"WebSocket error: {error}")
        self.reconnect()
    
    def on_close(self, ws, close_status_code, close_msg):
        # logger.info("WebSocket connection closed.")
        if not self.shutting_down and not self.reconnecting:
            self.reconnect()

    def subscribe_to_topic(self, symbol, topic_type):
        if not self.ws or not self.ws.sock or not self.ws.sock.connected:
            # logger.error("WebSocket is not connected.")
            return
        # logger.info(f"Subscribing to topic: {topic_type} for symbol: {symbol}")
        if not self.ws:
            # logger.warning(f"Trying to Subscribe to topic: {topic_type} for symbol: {symbol}, but but WebSocket is not initialized")
            return

        topic = self.topics.get(topic_type)
        if not topic:
            return
        subscribe_message = json.dumps({
            "id": str(int(time.time() * 1000)),
            "type": "subscribe",
            "topic": topic.format(symbol),
            "response": True
        })
        self.ws.send(subscribe_message)

    def unsubscribe_from_topic(self, symbol, topic_type):
        # logger.info(f"Unsubscribing to topic: {topic_type} for symbol: {symbol}")
        if not self.ws:
            # logger.warning(f"Trying to unsubscribe from topic {topic_type} for symbol {symbol} but WebSocket is not initialized.")
            return

        topic = self.topics.get(topic_type)
        if not topic:
            print("Unknown topic type:", topic_type)
            return

        unsubscribe_message = json.dumps({
            "id": str(int(time.time() * 1000)),
            "type": "unsubscribe",
            "topic": topic.format(symbol),
            "response": True
        })
        self.ws.send(unsubscribe_message)

    def ws_priceData(self, message_data):
        symbol = message_data.get('topic').split(":")[-1]
        data = message_data.get('data')
        if symbol and data:
            price_data = {
                "timestamp": data.get("time"), 
                "price": data.get("price")
            }
            self.update_latest_price_data(symbol, price_data)

            timestamp_datetime = datetime.fromtimestamp(price_data["timestamp"] / 1000.0, tz=timezone.utc)
            formatted_timestamp = timestamp_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def update_latest_price_data(self, symbol, price_data):
        with self.data_lock:
            self.latest_price_data[symbol] = price_data

    def get_latest_price_data(self, symbol):
        with self.data_lock:
            return self.latest_price_data.get(symbol, {"timestamp": None, "price": None})
    
    def send_ping(self):
        logging.DEBUG("Senden eines Ping")
        if self.ws_connected():
            self.last_ping_time = time.time()
            ping_message = json.dumps({"id": str(self.last_ping_time), "type": "ping"})
            self.ws.send(ping_message)
        else:
            logging.WARNING("try's to ping, but ws is not connected")

    def check_pong_received(self):
        if self.last_ping_time > self.last_pong_time:
            logging.WARNING("Pong not recieved, starts reconnecting function...")
            self.reconnect()

    def ping_thread(self):
        while not self.shutting_down:
            time.sleep(self.ping_interval)
            self.send_ping()

    def reconnect(self):
        if self.reconnecting:
            return
        self.reconnecting = True
        try:
            if self.ws:
                self.ws.close()
            self.request_token(private=False)
            self.reconnecting = False
        except Exception as e:
            # logger.error(f"Reconnection failed: {e}")
            self.reconnecting = False 

    def ws_connected(self):
        return self.ws is not None and self.ws.sock and self.ws.sock.connected

    def close_connection(self):
        if self.ws:
            self.ws.close()

    def shutdown_handler(self):
        self.shutting_down = True
        self.close_connection()


if __name__ == "__main__":
    client = KucoinWebSocketClient("JansConfig.ini")  # Initialisierung des Clients
    client.request_token(private=False)  # Anfrage eines Tokens
    time.sleep(3)  # Kurze Pause, um die Initialisierung abzuschließen

    symbol = "BTC-USDT"  # Symbol, für das Ticker-Daten abonniert werden sollen
    client.subscribe_to_topic(symbol, "ticker")  # Abonnieren von Ticker-Daten

    # Starte die kontinuierliche Aktualisierung
    time.sleep(3)  # Warte einen Moment, um Daten zu erhalten


    client.unsubscribe_from_topic(symbol, "ticker")  # Abonnement beenden
    client.shutdown_handler()  # Schließen der Verbindung und Beenden
