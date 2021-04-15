from telegram.ext import Updater, CommandHandler
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     level=logging.INFO)

import requests
import re
import random

PORT = int(os.environ.get('PORT', 5000))
mytoken = ""

def get_url():
    fall_pic_url = "https://drive.google.com/file/d/13RJGrN9xWbj2ryEsI6yqw2-9wHFGXxN8/view?usp=sharing"
    return fall_pic_url

def updates(update, context):
    url = get_url()
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=url)
    context.bot.send_message(chat_id=update.effective_chat.id, 
    text = "A fall has occured! Subject is at the study room.")


    
   
    

    

def main():
    updater = Updater(token='1666403358:AAHJ-d5KndPY5LHoV4YTwz9VBC8Dl4OF-eg', use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('updates', updates))
    updater.start_polling()
    
    # link webhook to heroku server
    # updater.start_webhook(listen="0.0.0.0",
    #                       port=int(PORT),
    #                       url_path=mytoken)
    # updater.bot.setWebhook('' + mytoken)

    updater.idle()

if __name__ == '__main__':
    main()

