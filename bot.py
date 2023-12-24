import os

import discord
from discord.ext import commands
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)

load_dotenv()

logging.info("init discord client")
TOKEN = os.getenv("DISCORD_TOKEN")
intents = discord.Intents().all()
bot = commands.Bot(command_prefix='!', intents=intents)

@bot.event
async def on_ready():
    logging.info(f'{bot.user} has connected to discord')

@bot.command(name="list")
async def raptor_list(ctx):
    await ctx.send("No list yet. Add one with '!hatch NAME'")

bot.run(TOKEN)
