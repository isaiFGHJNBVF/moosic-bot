import asyncio
import base64
import logging
import os
import re
import time
from collections import deque
from typing import Optional

import aiohttp
import discord
import yt_dlp
from discord import app_commands
from discord.ext import commands

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('MusicBot')

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TOKEN = os.environ.get('BOT_TOKEN_DISCORD')
if not TOKEN:
    raise RuntimeError('BOT_TOKEN_DISCORD environment variable is not set.')

SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', '')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '')

YDL_OPTIONS = {
    'format': 'bestaudio/best',
    'noplaylist': False,
    'quiet': True,
    'no_warnings': True,
    'default_search': 'ytsearch',
    'source_address': '0.0.0.0',
    'cookiefile': None,
    'postprocessors': [],
    'extract_flat': False,
    'skip_download': True,
    'ignoreerrors': True,
}

FFMPEG_OPTIONS = {
    'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5',
    'options': '-vn',
}

SPOTIFY_RE = re.compile(
    r'https?://open\.spotify\.com/(track|album|playlist)/([A-Za-z0-9]+)'
)


def is_url(text: str) -> bool:
    return text.startswith('http://') or text.startswith('https://')


# ---------------------------------------------------------------------------
# Spotify API client (Client Credentials flow — no user login needed)
# ---------------------------------------------------------------------------

class SpotifyClient:
    TOKEN_URL = 'https://accounts.spotify.com/api/token'
    API_BASE  = 'https://api.spotify.com/v1'

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self._token: Optional[str] = None
        self._token_expires: float = 0.0
        self._lock = asyncio.Lock()

    def _b64(self) -> str:
        return base64.b64encode(f'{self.client_id}:{self.client_secret}'.encode()).decode()

    async def _refresh(self, session: aiohttp.ClientSession):
        async with self._lock:
            if self._token and time.time() < self._token_expires - 30:
                return
            headers = {
                'Authorization': f'Basic {self._b64()}',
                'Content-Type': 'application/x-www-form-urlencoded',
            }
            async with session.post(self.TOKEN_URL, headers=headers,
                                    data={'grant_type': 'client_credentials'},
                                    timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    raise RuntimeError(f'Spotify token error: {r.status}')
                body = await r.json()
                self._token = body['access_token']
                self._token_expires = time.time() + body.get('expires_in', 3600)

    async def _get(self, path: str, session: aiohttp.ClientSession) -> dict:
        await self._refresh(session)
        headers = {'Authorization': f'Bearer {self._token}'}
        async with session.get(f'{self.API_BASE}{path}', headers=headers,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            r.raise_for_status()
            return await r.json()

    async def get_track(self, track_id: str) -> dict:
        async with aiohttp.ClientSession() as s:
            return await self._get(f'/tracks/{track_id}', s)

    async def get_album_tracks(self, album_id: str) -> list[dict]:
        async with aiohttp.ClientSession() as s:
            album = await self._get(f'/albums/{album_id}', s)
            return [
                {'title': f'{t["name"]} - {", ".join(a["name"] for a in t["artists"])}'}
                for t in album.get('tracks', {}).get('items', [])
            ]

    async def get_playlist_tracks(self, playlist_id: str) -> list[dict]:
        tracks, offset = [], 0
        async with aiohttp.ClientSession() as s:
            while True:
                data = await self._get(f'/playlists/{playlist_id}/tracks?limit=50&offset={offset}', s)
                items = data.get('items', [])
                if not items:
                    break
                for item in items:
                    t = item.get('track')
                    if t:
                        artists = ', '.join(a['name'] for a in t.get('artists', []))
                        tracks.append({'title': f'{t["name"]} - {artists}'})
                if not data.get('next'):
                    break
                offset += 50
        return tracks


_spotify: Optional[SpotifyClient] = None
if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
    _spotify = SpotifyClient(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
    logger.info('Spotify client initialized.')
else:
    logger.warning('Spotify credentials not set — Spotify links will use oembed fallback.')


# ---------------------------------------------------------------------------
# Song
# ---------------------------------------------------------------------------

class Song:
    __slots__ = ('title', 'url', 'stream_url', 'duration', 'thumbnail', 'requester')

    def __init__(self, title, url, stream_url, duration, thumbnail, requester):
        self.title = title
        self.url = url
        self.stream_url = stream_url
        self.duration = duration
        self.thumbnail = thumbnail
        self.requester = requester

    @property
    def duration_str(self) -> str:
        if not self.duration:
            return 'Live'
        m, s = divmod(self.duration, 60)
        h, m = divmod(m, 60)
        return f'{h}:{m:02d}:{s:02d}' if h else f'{m}:{s:02d}'

    def embed(self, title: str = 'Now Playing') -> discord.Embed:
        e = discord.Embed(title=title, description=f'[{self.title}]({self.url})',
                          color=discord.Color.blurple())
        if self.thumbnail:
            e.set_thumbnail(url=self.thumbnail)
        e.add_field(name='Duration', value=self.duration_str, inline=True)
        e.add_field(name='Requested by', value=self.requester.mention, inline=True)
        return e


# ---------------------------------------------------------------------------
# Guild music state (queue, loop, voice client per server)
# ---------------------------------------------------------------------------

class GuildMusicState:
    def __init__(self, bot, guild):
        self.bot = bot
        self.guild = guild
        self.voice_client: Optional[discord.VoiceClient] = None
        self.queue: deque[Song] = deque()
        self.current: Optional[Song] = None
        self.loop_current = False
        self.loop_queue = False
        self._play_next_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

    @property
    def is_playing(self):
        return self.voice_client is not None and self.voice_client.is_playing()

    @property
    def is_paused(self):
        return self.voice_client is not None and self.voice_client.is_paused()

    def _after_play(self, error):
        if error:
            logger.error(f'Playback error: {error}')
        self.bot.loop.call_soon_threadsafe(self._play_next_event.set)

    async def player_loop(self):
        await self.bot.wait_until_ready()
        while True:
            self._play_next_event.clear()
            if not self.loop_current:
                if self.loop_queue and self.current:
                    self.queue.append(self.current)
                if self.queue:
                    self.current = self.queue.popleft()
                else:
                    await asyncio.sleep(180)
                    if not self.queue and not self.is_playing:
                        if self.voice_client and self.voice_client.is_connected():
                            await self.voice_client.disconnect()
                        self.current = None
                    return
            if not self.current:
                return
            source = discord.PCMVolumeTransformer(
                discord.FFmpegPCMAudio(self.current.stream_url, **FFMPEG_OPTIONS), volume=1.0
            )
            if self.voice_client and self.voice_client.is_connected():
                self.voice_client.play(source, after=self._after_play)
            else:
                return
            await self._play_next_event.wait()

    def skip(self):
        if self.voice_client and (self.voice_client.is_playing() or self.voice_client.is_paused()):
            self.loop_current = False
            self.voice_client.stop()

    def pause(self):
        if self.voice_client and self.voice_client.is_playing():
            self.voice_client.pause()

    def resume(self):
        if self.voice_client and self.voice_client.is_paused():
            self.voice_client.resume()


# ---------------------------------------------------------------------------
# Spotify helpers
# ---------------------------------------------------------------------------

async def _spotify_queries(kind: str, spotify_id: str) -> list[str]:
    if not _spotify:
        return []
    try:
        if kind == 'track':
            data = await _spotify.get_track(spotify_id)
            artists = ', '.join(a['name'] for a in data.get('artists', []))
            return [f'ytsearch:{data["name"]} {artists}']
        elif kind == 'album':
            return [f'ytsearch:{t["title"]}' for t in await _spotify.get_album_tracks(spotify_id)]
        elif kind == 'playlist':
            return [f'ytsearch:{t["title"]}' for t in await _spotify.get_playlist_tracks(spotify_id)]
    except Exception as e:
        logger.error(f'Spotify error ({kind}/{spotify_id}): {e}')
    return []


async def _oembed_fallback(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f'https://open.spotify.com/oembed?url={url}',
                             timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status == 200:
                    data = await r.json()
                    title = data.get('title', '')
                    if title:
                        return f'ytsearch:{title}'
    except Exception as e:
        logger.warning(f'Spotify oembed fallback failed: {e}')
    return url


async def _ydl_extract(query: str) -> Optional[dict]:
    def _run():
        with yt_dlp.YoutubeDL(YDL_OPTIONS) as ydl:
            try:
                return ydl.extract_info(query, download=False)
            except Exception as e:
                logger.error(f'yt-dlp error: {e}')
                return None
    return await asyncio.get_event_loop().run_in_executor(None, _run)


def _info_to_songs(info: dict, original: str, requester) -> list[Song]:
    entries = info.get('entries') if 'entries' in info else [info]
    songs = []
    for entry in (entries or []):
        if not entry:
            continue
        url = entry.get('webpage_url') or entry.get('url') or original
        stream_url = entry.get('url') or ''
        if not stream_url:
            for fmt in reversed(entry.get('formats') or []):
                if fmt.get('acodec') != 'none' and fmt.get('url'):
                    stream_url = fmt['url']
                    break
        if not stream_url:
            continue
        songs.append(Song(
            title=entry.get('title') or 'Unknown',
            url=url,
            stream_url=stream_url,
            duration=entry.get('duration') or 0,
            thumbnail=entry.get('thumbnail') or '',
            requester=requester,
        ))
    return songs


async def fetch_songs(query: str, requester) -> Optional[list[Song]]:
    original = query
    match = SPOTIFY_RE.match(query)
    if match:
        kind, sid = match.group(1), match.group(2)
        queries = await _spotify_queries(kind, sid)
        if not queries:
            queries = [await _oembed_fallback(query)]
    elif not is_url(query):
        queries = [f'ytsearch:{query}']
    else:
        queries = [query]

    songs = []
    for q in queries:
        info = await _ydl_extract(q)
        if info:
            songs.extend(_info_to_songs(info, original, requester))
    return songs if songs else None


# ---------------------------------------------------------------------------
# Now Playing buttons
# ---------------------------------------------------------------------------

class NowPlayingView(discord.ui.View):
    def __init__(self, state: GuildMusicState):
        super().__init__(timeout=None)
        self.state = state

    def _in_vc(self, interaction: discord.Interaction) -> bool:
        return (interaction.user.voice is not None
                and self.state.voice_client is not None
                and interaction.user.voice.channel == self.state.voice_client.channel)

    @discord.ui.button(label='⏭ Skip', style=discord.ButtonStyle.primary)
    async def skip_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._in_vc(interaction):
            await interaction.response.send_message('You must be in the same voice channel.', ephemeral=True)
            return
        self.state.skip()
        await interaction.response.send_message('⏭ Skipped!')

    @discord.ui.button(label='⏸ Pause', style=discord.ButtonStyle.secondary)
    async def pause_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._in_vc(interaction):
            await interaction.response.send_message('You must be in the same voice channel.', ephemeral=True)
            return
        if self.state.is_playing:
            self.state.pause()
            button.label = '▶ Resume'
            button.style = discord.ButtonStyle.success
            await interaction.response.edit_message(view=self)
        elif self.state.is_paused:
            self.state.resume()
            button.label = '⏸ Pause'
            button.style = discord.ButtonStyle.secondary
            await interaction.response.edit_message(view=self)
        else:
            await interaction.response.send_message('Nothing is playing.', ephemeral=True)

    @discord.ui.button(label='⏹ Stop', style=discord.ButtonStyle.danger)
    async def stop_btn(self, interaction: discord.Interaction, button: discord.ui.Button):
        if not self._in_vc(interaction):
            await interaction.response.send_message('You must be in the same voice channel.', ephemeral=True)
            return
        self.state.queue.clear()
        self.state.loop_current = False
        self.state.loop_queue = False
        self.state.skip()
        await interaction.response.send_message('⏹ Stopped and cleared the queue.')


# ---------------------------------------------------------------------------
# Music Cog (all commands)
# ---------------------------------------------------------------------------

class Music(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.states: dict[int, GuildMusicState] = {}

    def _state(self, guild) -> GuildMusicState:
        if guild.id not in self.states:
            self.states[guild.id] = GuildMusicState(self.bot, guild)
        return self.states[guild.id]

    async def _ensure_voice(self, ctx) -> bool:
        if not ctx.author.voice or not ctx.author.voice.channel:
            await ctx.send('You must be in a voice channel first.')
            return False
        return True

    async def _join_vc(self, ctx) -> Optional[discord.VoiceClient]:
        state = self._state(ctx.guild)
        channel = ctx.author.voice.channel
        guild_vc: Optional[discord.VoiceClient] = ctx.guild.voice_client

        if guild_vc and guild_vc.is_connected():
            state.voice_client = guild_vc
            if guild_vc.channel.id != channel.id:
                await guild_vc.move_to(channel)
            return guild_vc

        for vc in [guild_vc, state.voice_client]:
            if vc:
                try:
                    await vc.disconnect(force=True)
                except Exception:
                    pass
        state.voice_client = None

        try:
            vc = await channel.connect(timeout=30.0, reconnect=True)
            state.voice_client = vc
            return vc
        except Exception as e:
            logger.error(f'Voice connect error: {e}')
            try:
                if ctx.interaction and not ctx.interaction.response.is_done():
                    await ctx.interaction.response.send_message(f'Failed to join: {e}')
                else:
                    await ctx.send(f'Failed to join: {e}')
            except Exception:
                pass
            return None

    async def _send(self, ctx, *args, **kwargs):
        if ctx.interaction:
            await ctx.interaction.followup.send(*args, **kwargs)
        else:
            await ctx.send(*args, **kwargs)

    # ── join ────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='join', description='Join your current voice channel.')
    async def join(self, ctx):
        if not await self._ensure_voice(ctx):
            return
        if ctx.interaction and not ctx.interaction.response.is_done():
            await ctx.interaction.response.defer(thinking=True)
        vc = await self._join_vc(ctx)
        if vc:
            await self._send(ctx, f'Joined **{vc.channel.name}**.')

    # ── leave ───────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='leave', description='Leave the voice channel and clear the queue.')
    async def leave(self, ctx):
        state = self._state(ctx.guild)
        if not state.voice_client or not state.voice_client.is_connected():
            await ctx.send('I am not in a voice channel.')
            return
        state.queue.clear()
        state.loop_current = False
        state.loop_queue = False
        if state._task:
            state._task.cancel()
        await state.voice_client.disconnect()
        state.voice_client = None
        state.current = None
        await ctx.send('Left the voice channel and cleared the queue.')

    # ── play ────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='play', description='Play a song by name or URL.')
    @app_commands.describe(query='Song name or URL (YouTube, Spotify, SoundCloud, etc.)')
    async def play(self, ctx, *, query: str):
        if not await self._ensure_voice(ctx):
            return
        if ctx.interaction and not ctx.interaction.response.is_done():
            await ctx.interaction.response.defer(thinking=True)
        else:
            await ctx.typing()

        vc = await self._join_vc(ctx)
        if not vc:
            return

        state = self._state(ctx.guild)
        songs = await fetch_songs(query, ctx.author)
        if not songs:
            await self._send(ctx, 'Could not find or stream that song. Try a different query.')
            return

        was_empty = not state.queue and state.current is None
        state.queue.extend(songs)

        if len(songs) > 1:
            embed = discord.Embed(title='Playlist Added',
                                  description=f'Added **{len(songs)}** songs to the queue.',
                                  color=discord.Color.green())
            await self._send(ctx, embed=embed)
        else:
            action = 'Added to Queue' if (not was_empty and state.current) else 'Now Playing'
            await self._send(ctx, embed=songs[0].embed(title=action), view=NowPlayingView(state))

        if not state._task or state._task.done():
            if not state.is_playing and not state.is_paused:
                state._task = self.bot.loop.create_task(state.player_loop())

    # ── skip ────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='skip', description='Skip the current song.')
    async def skip(self, ctx):
        state = self._state(ctx.guild)
        if not state.voice_client or not (state.is_playing or state.is_paused):
            await ctx.send('Nothing is currently playing.')
            return
        title = state.current.title if state.current else 'current song'
        state.skip()
        embed = discord.Embed(title='⏭ Skipped', description=f'Skipped **{title}**.',
                              color=discord.Color.orange())
        await ctx.send(embed=embed, view=NowPlayingView(state))

    # ── pause ───────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='pause', description='Pause the current song.')
    async def pause(self, ctx):
        state = self._state(ctx.guild)
        if not state.voice_client or not state.is_playing:
            await ctx.send('Nothing is currently playing.')
            return
        state.pause()
        embed = discord.Embed(title='⏸ Paused',
                              description=f'Paused **{state.current.title}**. Use `/resume` or `$$resume` to continue.',
                              color=discord.Color.yellow())
        await ctx.send(embed=embed, view=NowPlayingView(state))

    # ── resume ──────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='resume', description='Resume the paused song.')
    async def resume(self, ctx):
        state = self._state(ctx.guild)
        if not state.voice_client or not state.is_paused:
            await ctx.send('The music is not paused.')
            return
        state.resume()
        embed = discord.Embed(title='▶ Resumed',
                              description=f'Resumed **{state.current.title}**.',
                              color=discord.Color.green())
        await ctx.send(embed=embed, view=NowPlayingView(state))

    # ── list ────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='list', description='Show the current song queue.')
    async def list(self, ctx):
        state = self._state(ctx.guild)
        embed = discord.Embed(title='Song Queue', color=discord.Color.blurple())
        if state.current:
            status = '▶ Playing' if state.is_playing else '⏸ Paused'
            embed.add_field(name=status,
                            value=f'[{state.current.title}]({state.current.url}) — `{state.current.duration_str}` — {state.current.requester.mention}',
                            inline=False)
        else:
            embed.add_field(name='Currently Playing', value='Nothing', inline=False)
        if state.queue:
            lines = [f'`{i}.` [{s.title}]({s.url}) — `{s.duration_str}` — {s.requester.mention}'
                     for i, s in enumerate(list(state.queue)[:20], 1)]
            if len(state.queue) > 20:
                lines.append(f'_...and {len(state.queue) - 20} more_')
            embed.add_field(name=f'Queue ({len(state.queue)} songs)', value='\n'.join(lines), inline=False)
        else:
            embed.add_field(name='Queue', value='Empty', inline=False)
        footer = []
        if state.loop_current:
            footer.append('🔂 Looping current song')
        if state.loop_queue:
            footer.append('🔁 Looping whole playlist')
        if footer:
            embed.set_footer(text=' | '.join(footer))
        await ctx.send(embed=embed)

    # ── loop ────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='loop', description='Set loop mode for current song and/or queue.')
    @app_commands.describe(current='Loop the current song (True/False)',
                           playlist='Loop the whole queue (True/False)')
    async def loop(self, ctx, current: bool, playlist: bool):
        state = self._state(ctx.guild)
        state.loop_current = current
        state.loop_queue = playlist
        parts = []
        if current:
            parts.append('🔂 Looping **current song**')
        if playlist:
            parts.append('🔁 Looping **whole playlist**')
        if not parts:
            parts = ['Loop disabled']
        embed = discord.Embed(title='Loop Settings Updated', description='\n'.join(parts),
                              color=discord.Color.blurple())
        await ctx.send(embed=embed)

    # ── nowplaying ──────────────────────────────────────────────────────────

    @commands.hybrid_command(name='nowplaying', aliases=['np'], description='Show what is currently playing.')
    async def nowplaying(self, ctx):
        state = self._state(ctx.guild)
        if not state.current:
            await ctx.send('Nothing is currently playing.')
            return
        status = '▶ Now Playing' if state.is_playing else '⏸ Paused'
        await ctx.send(embed=state.current.embed(title=status), view=NowPlayingView(state))

    # ── clear ───────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='clear', description='Clear the entire queue.')
    async def clear(self, ctx):
        self._state(ctx.guild).queue.clear()
        await ctx.send('Queue cleared.')

    # ── volume ──────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='volume', description='Set playback volume (0–200).')
    @app_commands.describe(level='Volume level from 0 to 200')
    async def volume(self, ctx, level: int):
        state = self._state(ctx.guild)
        if not state.voice_client or not (state.is_playing or state.is_paused):
            await ctx.send('Nothing is currently playing.')
            return
        if not 0 <= level <= 200:
            await ctx.send('Volume must be between 0 and 200.')
            return
        if isinstance(state.voice_client.source, discord.PCMVolumeTransformer):
            state.voice_client.source.volume = level / 100
        await ctx.send(f'Volume set to **{level}%**.')

    # ── help ────────────────────────────────────────────────────────────────

    @commands.hybrid_command(name='help', description='Show all music bot commands.')
    async def help(self, ctx):
        spotify_status = '✅ Connected' if _spotify else '⚠️ Not configured'
        embed = discord.Embed(
            title='Music Bot — Commands',
            description=f'Use `/command` (slash) or `$$command` (prefix)\nSpotify API: {spotify_status}',
            color=discord.Color.blurple(),
        )
        for name, desc in [
            ('join',                    'Join your voice channel'),
            ('leave',                   'Leave and clear the queue'),
            ('play <query>',            'Play by name or URL — YouTube, Spotify, SoundCloud, etc.'),
            ('skip',                    'Skip the current song'),
            ('pause',                   'Pause the current song'),
            ('resume',                  'Resume the paused song'),
            ('list',                    'Show the song queue'),
            ('loop <current> <queue>',  'Set loop modes — True or False each'),
            ('nowplaying  |  np',       'Show what is currently playing'),
            ('volume <0–200>',          'Set the playback volume'),
            ('clear',                   'Clear the queue'),
            ('help',                    'Show this message'),
        ]:
            embed.add_field(name=f'/{name}  |  $${name}', value=desc, inline=False)
        embed.set_footer(text='Tip: Skip / Pause / Stop buttons appear on Now Playing messages.')
        await ctx.send(embed=embed)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

intents = discord.Intents.default()
intents.message_content = True
intents.voice_states = True


class MusicBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix='$$', intents=intents, help_command=None)

    async def setup_hook(self):
        await self.add_cog(Music(self))
        logger.info('Music cog loaded.')
        await self.tree.sync()
        logger.info('Slash commands synced globally.')

    async def on_ready(self):
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        await self.change_presence(
            activity=discord.Activity(type=discord.ActivityType.listening, name='/play | $$play')
        )

    async def on_command_error(self, ctx, error):
        if isinstance(error, commands.CommandNotFound):
            return
        if isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f'Missing argument: `{error.param.name}`')
            return
        if isinstance(error, commands.CheckFailure):
            await ctx.send(str(error))
            return
        logger.error(f'Command error in {ctx.command}: {error}', exc_info=error)


async def main():
    bot = MusicBot()
    async with bot:
        await bot.start(TOKEN)


if __name__ == '__main__':
    asyncio.run(main())
