import os
import sys
import configparser
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
import logging
import asyncio


base_path = os.path.dirname(os.path.realpath(__file__))
crypto_bot_path = os.path.dirname(base_path)
Python_path = os.path.dirname(crypto_bot_path)
kucoin_api_path = os.path.join(crypto_bot_path, "Kucoin API")
config_path = os.path.join(crypto_bot_path, "Config")
Trading_bot_path = os.path.dirname(Python_path)
Trading_path = os.path.join(Trading_bot_path, "Trading")
ledger_path = os.path.join(Trading_path, "trading_ledger.json")
utils_path = os.path.join(Python_path, "Tools")
mo_utils_path = os.path.join(utils_path, "mo_utils")
print(os.listdir(kucoin_api_path))



sys.path.append(utils_path)
sys.path.append(mo_utils_path)

sys.path.append(crypto_bot_path)
sys.path.append(kucoin_api_path)
sys.path.append(config_path)
sys.path.append(Trading_path)

try:
    from KucoinTrader import load_ledger_data
except Exception as error:
    print(error)

import mo_utils as utils



config = configparser.ConfigParser()
telegram_config_ini_path = os.path.join(config_path, "telegram_config.ini")
config.read(telegram_config_ini_path)

token = config.get("token_data", "token").strip('"')
login_password = config.get("token_data", "login_password").strip('"')
action_password = config.get("token_data", "action_password").strip('"')


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)


waiting_for_action_password = {}
waiting_for_login_password = set()
authenticated_users = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await request_login_password(update, context)

async def request_login_password(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    waiting_for_login_password.add(user_id)
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Enter Login Password:")

def auth_required(handler):
    async def check_auth(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user_id = update.message.from_user.id
        if user_id in authenticated_users:
            await handler(update, context)
        else:
            await request_login_password(update, context)
    return check_auth

async def check_password(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    if user_id in waiting_for_login_password:
        if update.message.text == login_password:
            authenticated_users[user_id] = True
            waiting_for_login_password.remove(user_id)
            await update.message.reply_text("Login successful. You can now use the bot. Waiting a Command:")
        else:
            await update.message.reply_text("Incorrect password. Access denied.")
    elif user_id in waiting_for_action_password:
        if update.message.text == action_password:
            await update.message.reply_text("Action password accepted.")
            waiting_for_action_password.remove(user_id)
        else:
            await update.message.reply_text("Incorrect password. Action canceled.")

@auth_required
async def open_positions(update: Update, context: ContextTypes.DEFAULT_TYPE):
    open_positions_data = {} #  add the funktion
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"open postions: {open_positions_data}")

@auth_required
async def ledger_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        ledger_info = load_ledger_data(filepath=ledger_path)
        if ledger_info is None:
            ledger_info = "No Ledger Information"
            raise ValueError("Data Empty or not callable")
    except Exception as error:
        ledger_info = f"Failed loading ledger_data. Error: {type(error).__name__}, {error}""Failed loading ledger_data. Error:\n", type(error).__name__

    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Ledger Data: {ledger_info}")

@auth_required
async def exit_all(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Enter Password:")
    waiting_for_action_password[user_id] = True

@auth_required
async def performance_daily(update: Update, context: ContextTypes.DEFAULT_TYPE):
    performance_daily_data = {} #  add the funktion
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"performance_daily_data: {performance_daily_data}")

@auth_required
async def performance_monthly(update: Update, context: ContextTypes.DEFAULT_TYPE):
    performance_monthly_data = {} #  add the funktion
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"performance_monthly: {performance_monthly_data}")

@auth_required
async def account_balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    account_balance_data = {} #  add the funktion
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"account_balance: {account_balance_data}")

if __name__ == '__main__':
    application = ApplicationBuilder().token(token).build()

    # Command handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CommandHandler('ledger', ledger_data))
    application.add_handler(CommandHandler('open_positions', open_positions))
    application.add_handler(CommandHandler('exit_all', exit_all))
    application.add_handler(CommandHandler('performance_daily', performance_daily))
    application.add_handler(CommandHandler('performance_monthly', performance_monthly))
    application.add_handler(CommandHandler('account_balance', account_balance))
    
    # Non-command message handler for checking passwords
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_password))

    # Start the bot
    application.run_polling()