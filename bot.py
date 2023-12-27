import os

import discord
from discord.ext import commands
from dotenv import load_dotenv
from modules import utils
from modules.raptorai.raptor import raptorai

load_dotenv()

logger = utils.get_logger("discord_bot")

logger.info('init discord client')
TOKEN = os.getenv('DISCORD_TOKEN')
intents = discord.Intents().all()
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

@client.event
async def on_ready():
    # await tree.sync(guild=discord.Object(id=Your guild id))
    logger.info(f'{client.user} has connected to discord')
    for guild in client.guilds:
        logger.info(f'Guild id:{guild.id}')
        await tree.sync(guild=guild)

# , guild=discord.Object(id=12417128931)) #Add the guild ids in which the slash command will appear. If it should be in all, remove the argument, but note that it will take some time (up to an hour) to register the command if it's for all guilds.
@tree.command(name='list', description='list the raptors you have trained', guild=discord.Object(id=1138385140674998312))
async def raptor_list(interaction):
    await interaction.response.defer()
    ret = await raptorai.list()
    await interaction.followup.send(ret)

@tree.command(name='feed', description='feed your raptor with some yumi content', guild=discord.Object(id=1138385140674998312))
async def raptor_feed(interaction, url: str):
    await interaction.response.defer()
    ret = await raptorai.feed(url)
    response = f'{ret} ({url})'
    await interaction.followup.send(ret)

@tree.command(name='ask', description='ask your raptor', guild=discord.Object(id=1138385140674998312))
async def raptor_feed(interaction, question: str):
    await interaction.response.defer()
    ret = await raptorai.ask(question)
    response = f'Question: {question}\nAnswer:\n{ret}'
    await interaction.followup.send(response)

@tree.command(name='kill', description='kill your raptor :(', guild=discord.Object(id=1138385140674998312))
async def raptor_kill(interaction):
    await interaction.response.defer()
    ret = await raptorai.kill()
    await interaction.followup.send('Raptor hunted sir')

@tree.command(name='hatch', description='hatch a new raptor \o/', guild=discord.Object(id=1138385140674998312))
async def raptor_hatch(interaction):
    await interaction.response.defer()
    ret = await raptorai.hatch()
    await interaction.followup.send(f'A warm welcome to your new raptor {ret}')


client.run(TOKEN)
