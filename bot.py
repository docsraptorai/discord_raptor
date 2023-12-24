import os

import discord
from discord.ext import commands
from dotenv import load_dotenv
import logging
from modules.raptorai import raptor

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
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

@client.event
async def on_ready():
    # await tree.sync(guild=discord.Object(id=Your guild id))
    logging.info(f'{client.user} has connected to discord')
    for guild in client.guilds:
        logging.info(f'Guild id:{guild.id}')
        await tree.sync(guild=guild)

# , guild=discord.Object(id=12417128931)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
@tree.command(name="list", description="list the raptors you have trained", guild=discord.Object(id=1138385140674998312))
async def raptor_list(interaction):
    await interaction.response.send_message(raptor.r_list())


client.run(TOKEN)
