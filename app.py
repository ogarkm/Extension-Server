import os
import importlib.util
import json
import socket
import threading
import uvicorn
import time
import asyncio
import uuid
from urllib.parse import urlparse
from contextlib import asynccontextmanager
import re
import shutil
import subprocess
import signal
import traceback
import zipfile
import io
from fastapi import FastAPI, HTTPException, Query, Body, status, UploadFile, File, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import httpx
import io
from fastapi.responses import StreamingResponse, JSONResponse, Response
from zeroconf import ServiceInfo, Zeroconf
from fpdf import FPDF

# --- Global Event for Animation Control ---
server_ready_event = threading.Event()

def animate_loading(stop_event: threading.Event):
    """Displays a loading animation in the console until the stop_event is set."""
    animation_chars = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
    idx = 0
    print("🎬 Starting Animex Extension Server...", end="", flush=True)
    while not stop_event.is_set():
        char = animation_chars[idx % len(animation_chars)]
        print(f" {char}", end="\r", flush=True)
        idx += 1
        time.sleep(0.08)
    time.sleep(2)  # Give a moment to clear the last character
    print(" " * 5, end="\r", flush=True)
    print("📺 Animex Extension Server is ready for anime streaming!")
    print("Press CTRL+C to stop the Animex server\n")

# --- Utility Functions ---
def natural_sort_key(s):
    """A key function for natural sorting of strings containing numbers."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

# --- Zeroconf Service Registration ---
zeroconf = Zeroconf()
service_info = None

def get_local_ip():
    """Finds the local IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP


def register_service():
    """Registers this API as a service on the local network."""
    global service_info
    try:
        host_ip = get_local_ip()
        host_name = socket.gethostname()
        port = 7275
        service_info = ServiceInfo(
            "_http._tcp.local.",
            f"Animex @ {host_name}._http._tcp.local.",
            addresses=[socket.inet_aton(host_ip)],
            port=port,
            properties={'app': 'animex-extension-api'},
            server=f"{host_name}.local.",
        )
        print(f"Registering service '{service_info.name}' on {host_ip}:{port}")
        zeroconf.register_service(service_info)
        print("Service registration completed")
    except Exception as e:
        print(f"Failed to register Zeroconf service: {e}")

async def unregister_service():
    """Unregisters the service from the network on shutdown."""
    if service_info:
        print(f"Unregistering service '{service_info.name}'")
        zeroconf.close()
        
        
# --- Caching Setup ---
import hashlib
import time

DATA_DIR = "data"

CACHE_DIR_JIKAN = os.path.join(DATA_DIR, "cache", "jikan")
os.makedirs(CACHE_DIR_JIKAN, exist_ok=True)

# In-memory cache for the large Anime DB JSON to prevent re-downloading
MEMORY_CACHE = {
    "anime_db": None,
    "anime_db_timestamp": 0
}

def get_cache_key(url: str) -> str:
    """Generates a safe filename from a URL."""
    return hashlib.md5(url.encode('utf-8')).hexdigest() + ".json"

def load_cache(url: str):
    """Loads data from disk if valid."""
    filepath = os.path.join(CACHE_DIR_JIKAN, get_cache_key(url))
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If marked permanent, return immediately
            if data.get("_is_permanent", False):
                return data["payload"]
            
            # If expired (24 hours = 86400 seconds), return None
            if time.time() - data["_timestamp"] > 86400:
                return None
                
            return data["payload"]
        except Exception:
            return None
    return None

def save_cache(url: str, payload: dict):
    """Saves data to disk. Checks if anime/manga is finished to mark permanent."""
    filepath = os.path.join(CACHE_DIR_JIKAN, get_cache_key(url))
    
    is_permanent = False
    
    # Check Jikan 'status' field to see if we can cache forever
    # Supports both singular resource ('data': {...}) and lists ('data': [...])
    data_content = payload.get("data")
    
    if isinstance(data_content, dict):
        status = data_content.get("status", "")
        if status in ["Finished Airing", "Finished"]:
            is_permanent = True
            
    cache_obj = {
        "payload": payload,
        "_timestamp": time.time(),
        "_is_permanent": is_permanent
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cache_obj, f)
    except Exception as e:
        print(f"Failed to save cache: {e}")

# --- Module & Extension Loading ---
MODULES_DIR = "modules"
EXTENSIONS_DIR = "extensions"
loaded_modules = {}
module_states = {}
loaded_extensions = {}
def load_modules():
    if not os.path.exists(MODULES_DIR):
        os.makedirs(MODULES_DIR)
    
    module_files = sorted([f for f in os.listdir(MODULES_DIR) if f.endswith(".module")])
    
    for filename in module_files:
        module_name = filename.split(".")[0]
        if module_name in loaded_modules:
            continue
        
        try:
            with open(os.path.join(MODULES_DIR, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                meta_str, _, code_str = content.partition("\n---\n")
                
                spec = importlib.util.spec_from_loader(module_name, loader=None)
                module = importlib.util.module_from_spec(spec)
                
                exec(code_str, module.__dict__)
                
                module_info = json.loads(meta_str)
                module_info['id'] = module_name
                loaded_modules[module_name] = {
                    "info": module_info,
                    "instance": module,
                }
                module_states[module_name] = True
                print(f"Successfully loaded module: {module_info.get('name', module_name)}")
        except Exception as e:
            print(f"Failed to load module {filename}: {e}")
            
            
def load_extensions(app: FastAPI):
    if not os.path.exists(EXTENSIONS_DIR):
        return

    for ext_name in os.listdir(EXTENSIONS_DIR):
        ext_path = os.path.join(EXTENSIONS_DIR, ext_name)
        if not os.path.isdir(ext_path):
            continue

        package_json_path = os.path.join(ext_path, "package.json")
        if not os.path.exists(package_json_path):
            continue

        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                ext_meta = json.load(f)
            
            main_file = ext_meta.get("main", "extension.extn")
            ext_file_path = os.path.join(ext_path, main_file)

            module_name = f"extensions.{ext_name}.{main_file.split('.')[0]}"
            spec = importlib.util.spec_from_file_location(module_name, ext_file_path)
            ext_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ext_module)
            
            ext_module.EXT_PATH = os.path.abspath(ext_path)

            loaded_extensions[ext_name] = {
                "info": ext_meta,
                "instance": ext_module,
                "process": None,
                "server_url": None
            }

            if "port" in ext_meta and "start_command" in ext_meta:
                ext_port = ext_meta["port"]
                ext_start_command = ext_meta["start_command"]
                ext_server_url = f"http://127.0.0.1:{ext_port}"

                print(f"Starting extension '{ext_name}' server in the background.")
                
                process = subprocess.Popen(
                    ext_start_command,
                    shell=True,
                    preexec_fn=os.setsid,
                    cwd=ext_path
                )
                
                loaded_extensions[ext_name]["process"] = process
                loaded_extensions[ext_name]["server_url"] = ext_server_url
                print(f"Extension '{ext_name}' server process started with PID: {process.pid}.")

            if "static_folder" in ext_meta:
                static_path = os.path.join(ext_path, ext_meta["static_folder"])
                if os.path.isdir(static_path):
                    app.mount(f"/ext/{ext_name}/static", StaticFiles(directory=static_path), name=f"ext_{ext_name}_static")
                    print(f"Mounted static folder for '{ext_name}' at /ext/{ext_name}/static")

            print(f"Successfully loaded extension logic: {ext_meta.get('name', ext_name)}")

        except Exception as e:
            
            print(f"Failed to load extension {ext_name}: {e}")
            traceback.print_exc()

# --- Profile Management ---

AVATARS_DIR = os.path.join(DATA_DIR, "avatars")
PROFILES_FILE = os.path.join(DATA_DIR, "profiles.json")

class ProfileDict(dict):
    def get(self, key, default=None):
        if key is None:
            if self:
                return next(iter(self.values()), None)
            return None
        return super().get(key, default)

profiles: ProfileDict = ProfileDict()
profiles_lock = threading.Lock()

class ProfileSettings(BaseModel):
    theme: str = "dark"
    pushNotifications: bool = False
    downloadQuality: str = "1080p"
    nsfw_enabled: bool = False
    module_preferences: Dict[str, List[str]] = Field(default_factory=dict)

class WatchedEpisode(BaseModel):
    watched_at: str
    season_number: int = 1
    state: str = "finished"  # 'ongoing' or 'finished'
    timestamp: int = 0

class WatchHistoryItem(BaseModel):
    title: str
    episodes: Dict[str, WatchedEpisode] = Field(default_factory=dict)
    last_watched: str

class Profile(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    avatar_url: str
    settings: ProfileSettings = Field(default_factory=ProfileSettings)
    watch_history: Dict[str, WatchHistoryItem] = Field(default_factory=dict)

class CreateProfileRequest(BaseModel):
    name: str
    
    
def save_profiles():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(AVATARS_DIR, exist_ok=True)
        with open(PROFILES_FILE, 'w', encoding='utf-8') as f:
            json.dump([p.dict() for p in profiles.values()], f, indent=4)
    except Exception as e:
        print(f"Error saving profiles: {e}")

def save_profiles_with_lock():
    with profiles_lock:
        save_profiles()

def load_profiles():
    global profiles
    with profiles_lock:
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
        try:
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                profiles_list = json.load(f)
                profiles = ProfileDict({p['id']: Profile(**p) for p in profiles_list})
                print(f"Loaded {len(profiles)} profiles from {PROFILES_FILE}")
        except (FileNotFoundError, json.JSONDecodeError):
            profiles = ProfileDict()
            print("No existing profiles file found or file is empty. Starting fresh.")
            save_profiles()
            print("Created an empty profiles file.")

# --- FastAPI App Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    load_modules()
    load_profiles()
    load_extensions(app)
    
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, register_service)
    
    server_ready_event.set()
    
    yield

    # --- Shutdown ---
    print("\nShutting down Animex Extension Server...")
    # Correctly iterate through loaded_extensions to find running processes
    for ext_name, ext_data in loaded_extensions.items():
        process = ext_data.get("process")
        if process and process.poll() is None:
            print(f"Terminating extension '{ext_name}' server (PID: {process.pid})...")
            try:
                # Terminate the entire process group started by the extension
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
                print(f"Extension '{ext_name}' server terminated successfully.")
            except (ProcessLookupError, OSError) as e:
                print(f"Could not terminate extension '{ext_name}' server (it may have already closed): {e}")
            except Exception as e:
                print(f"An unexpected error occurred while terminating extension '{ext_name}': {e}")

    await unregister_service()
    save_profiles_with_lock()
    print("Profiles saved.")
    print("Animex Extension Server shutdown complete.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Animex Extensions API",
    description="A modular API for fetching anime stream and download information.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MangaDex API Configuration ---
MANGADEX_API_URL = "https://api.mangadex.org"

# --- Endpoints (Omitted for brevity, no changes were made to them) ---
@app.get("/settings/add-ons", response_model=List[Dict[str, Any]])
async def get_add_on_settings():
    """Returns settings metadata from all loaded extensions."""
    settings = []
    for ext_name, ext_data in loaded_extensions.items():
        if "settings" in ext_data["info"]:
            settings.append({
                "extension_id": ext_name,
                **ext_data["info"]["settings"]
            })
    return settings


@app.get("/extensions", response_model=List[str])
async def get_extensions_list():
    """Returns a list of all loaded extension IDs."""
    return list(loaded_extensions.keys())


@app.post("/settings/add-on")
async def set_add_on_settings(settings_data: Dict[str, Any] = Body(...)):
    """Saves settings for a specific extension."""
    extension_id = settings_data.get("extension_id")
    if not extension_id or extension_id not in loaded_extensions:
        raise HTTPException(status_code=404, detail="Extension not found.")
    
    # Here you would typically save the settings to a file or database
    # For now, we'll just print them to the console
    print(f"Received settings for {extension_id}: {settings_data}")
    
    return {"status": "success", "extension_id": extension_id, "settings": settings_data}

@app.get("/ext/{ext_name}/info", response_model=Dict[str, Any])
async def get_extension_info(ext_name: str):
    """Returns metadata for a specific loaded extension."""
    ext_data = loaded_extensions.get(ext_name)
    if not ext_data:
        raise HTTPException(status_code=404, detail="Extension not found.")
    return ext_data["info"]

@app.get("/identify", include_in_schema=False)
def identify_server():
    return {"app": "Animex Extension API", "version": "1.0"}

@app.get("/status")
async def get_status():
    return {"status": "online"}


@app.get("/export/series/{mal_id}")
async def export_series_package(mal_id: int, type: str = Query(..., enum=["anime", "manga"])):
    """
    Creates and returns a zip file containing a folder with the series title,
    a meta.json file, and a poster.png.
    """
    # 1. Fetch series details from Jikan
    try:
        async with httpx.AsyncClient() as client:
            details_resp = await client.get(f"https://api.jikan.moe/v4/{type}/{mal_id}")
            details_resp.raise_for_status()
            series_data = details_resp.json().get("data", {})
            if not series_data:
                raise HTTPException(status_code=404, detail="Series not found on Jikan.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=f"Jikan API error: {e.response.text}")

    series_title = series_data.get("title_english") or series_data.get("title", f"series_{mal_id}")
    if not series_title and type == "manga":
        series_title = series_data.get("title", f"series_{mal_id}")
    safe_series_title = "".join(c for c in series_title if c.isalnum() or c in (' ', '_')).rstrip()

    poster_url = series_data.get("images", {}).get("jpg", {}).get("large_image_url")
    
    # 2. Fetch poster image
    poster_content = None
    if poster_url:
        try:
            async with httpx.AsyncClient() as client:
                poster_resp = await client.get(poster_url)
                poster_resp.raise_for_status()
                poster_content = poster_resp.content
        except httpx.RequestError:
            poster_content = None # Fail gracefully if poster can't be downloaded

    # 3. Create the package structure in a temporary directory
    temp_dir = f"temp_export_{uuid.uuid4()}"
    series_dir = os.path.join(temp_dir, safe_series_title)
    os.makedirs(series_dir, exist_ok=True)

    try:
        # Create meta.json
        meta_path = os.path.join(series_dir, "meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(series_data, f, indent=2)

        # Save poster.png
        if poster_content:
            poster_path = os.path.join(series_dir, "poster.png")
            with open(poster_path, 'wb') as f:
                f.write(poster_content)

        # 4. Create the zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(series_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
        
        zip_buffer.seek(0)

    finally:
        # 5. Clean up the temporary directory
        shutil.rmtree(temp_dir)

    # 6. Return the zip file
    zip_filename = f"{safe_series_title}.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=\"{zip_filename}\""}
    )



# --- Profile Management Endpoints ---

@app.get("/profiles", response_model=List[Profile])
async def get_all_profiles():
    """Retrieve all user profiles."""
    return sorted(list(profiles.values()), key=lambda p: p.name.lower())


def get_profile_setting(user_id: str, setting_name: str) -> any:
    """
    Retrieves a specific setting from a user's profile.

    Args:
    user_id (str): The ID of the user.
    setting_name (str): The name of the setting to retrieve.

    Returns:
    any: The value of the setting, or None if not found.
    """
    try:
        with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
            user_profile = next((profile for profile in profiles if profile['id'] == user_id), None)
            if user_profile:
                return user_profile.get('settings', {}).get(setting_name)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None

@app.post("/profiles", response_model=Profile, status_code=status.HTTP_201_CREATED)
async def create_profile(profile_req: CreateProfileRequest):
    """Create a new user profile."""
    if len(profiles) >= 10: # Limit number of profiles
        raise HTTPException(status_code=400, detail="Maximum number of profiles reached.")
    if not profile_req.name or len(profile_req.name) > 20:
        raise HTTPException(status_code=400, detail="Profile name must be between 1 and 20 characters.")

    initial = profile_req.name[0].upper()
    avatar = f"https://placehold.co/100/FF9500/FFFFFF?text={initial}"
    
    new_profile = Profile(name=profile_req.name.strip(), avatar_url=avatar)
    profiles[new_profile.id] = new_profile
    save_profiles()
    return new_profile

@app.put("/profiles/{profile_id}", response_model=Profile)
async def update_profile(profile_id: str, updated_profile: Profile):
    """Update an existing profile's name or settings."""
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    if profile_id != updated_profile.id:
        raise HTTPException(status_code=400, detail="Profile ID mismatch.")
    
    profiles[profile_id] = updated_profile
    save_profiles()
    return updated_profile

@app.delete("/profiles/{profile_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_profile(profile_id: str):
    """Delete a profile."""
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    
    del profiles[profile_id]
    save_profiles()
    return

@app.get("/profiles/{profile_id}", response_model=Profile)
async def get_profile(profile_id: str):
    """Get a specific profile by ID."""
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return profiles[profile_id]


@app.get("/profiles/{profile_id}/watch-history", response_model=Dict[str, WatchHistoryItem])
async def get_watch_history(profile_id: str):
    """Retrieve the watch history for a specific profile."""
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    profile = profiles.get(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found.")
    return getattr(profile, 'watch_history', {})


@app.patch("/profiles/{profile_id}", response_model=Profile)
async def patch_profile(profile_id: str, profile_data: dict):
    """Partially update an existing profile."""
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    
    current_profile = profiles[profile_id]
    
    # Update only the provided fields
    if "name" in profile_data:
        current_profile.name = profile_data["name"].strip()
    
    if "settings" in profile_data:
        current_profile.settings = ProfileSettings(**profile_data["settings"])
    
    profiles[profile_id] = current_profile
    save_profiles_with_lock()
    return current_profile


@app.patch("/profiles/{profile_id}/settings", response_model=Profile)
async def patch_profile_settings(profile_id: str, settings_data: ProfileSettings):
    """Partially update an existing profile's settings."""
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")
    
    current_profile = profiles[profile_id]
    current_profile.settings = settings_data # Directly assign the new settings
    
    profiles[profile_id] = current_profile
    save_profiles_with_lock()
    return current_profile


@app.patch("/profiles/{profile_id}/module-preferences", response_model=Profile)
async def patch_module_preferences(profile_id: str, module_prefs: Dict[str, List[str]] = Body(...)):
    """Updates the module preferences for a specific profile."""
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")

    current_profile = profiles[profile_id]
    current_profile.settings.module_preferences = module_prefs
    save_profiles_with_lock()
    return current_profile


@app.post("/profiles/{profile_id}/avatar", response_model=Profile)
async def upload_avatar(profile_id: str, avatar: UploadFile = File(...)):
    """Upload a new avatar for a profile."""
    if profile_id not in profiles:
        raise HTTPException(status_code=404, detail="Profile not found.")

    if not avatar.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        file_extension = avatar.filename.split('.')[-1]
        avatar_filename = f"{profile_id}.{file_extension}"
        avatar_path = os.path.join(AVATARS_DIR, avatar_filename)

        with open(avatar_path, "wb") as buffer:
            shutil.copyfileobj(avatar.file, buffer)

        # Update profile with the new avatar URL
        # The URL should be a relative path that the client can use
        avatar_url = f"/data/avatars/{avatar_filename}"
        profiles[profile_id].avatar_url = avatar_url
        save_profiles_with_lock()

        return profiles[profile_id]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload avatar: {e}")



# --- Module Endpoints ---

@app.get("/modules", response_model=List[Dict[str, Any]])
def get_modules_list():
    return [
        {**mod['info'], "enabled": module_states.get(name, False), "nsfw": mod['info'].get("nsfw", False)}
        for name, mod in loaded_modules.items()
    ]

@app.get("/modules/status")
def get_modules_status():
    """Returns the current enabled/disabled status of all modules."""
    return module_states

@app.get("/modules/toggle/{module_id}/{enable}")
def toggle_module_status(module_id: str, enable: bool):
    """Enable or disable a specific module."""
    if module_id not in loaded_modules:
        raise HTTPException(status_code=404, detail=f"Module '{module_id}' not found.")
    
    module_states[module_id] = enable
    status = "enabled" if enable else "disabled"
    print(f"Module '{module_id}' has been {status}.")
    return {"module_id": module_id, "status": status}


@app.get("/modules/streaming", response_model=List[Dict[str, Any]])
def get_streaming_modules():
    """Returns a list of all enabled ANIME_STREAMER modules."""
    streaming_modules = []
    for name, mod in loaded_modules.items():
        if module_states.get(name, False):
            module_info = mod.get("info", {})
            module_type = module_info.get("type")
            is_streamer = (isinstance(module_type, list) and "ANIME_STREAMER" in module_type) or \
                          (isinstance(module_type, str) and module_type == "ANIME_STREAMER")
            if is_streamer:
                streaming_modules.append({
                    "id": name,
                    "name": module_info.get("name", name),
                    "version": module_info.get("version", "N/A")
                })
    return streaming_modules


# --- Core Content Endpoints ---


def _get_cover_url_from_manga(manga: Dict[str, Any]) -> Optional[str]:
    """Helper to extract cover URL from a MangaDex manga object with included cover_art."""
    cover_rel = next((rel for rel in manga.get("relationships", []) if rel.get("type") == "cover_art"), None)
    if cover_rel:
        file_name = cover_rel.get("attributes", {}).get("fileName")
        if file_name:
            # Return a relative path to our new proxy endpoint
            return f"/mangadex/cover/{manga['id']}/{file_name}"
    return None

# --- MangaDex API Endpoints ---
@app.get("/mangadex/search")
async def search_mangadex(q: str, profile_id: Optional[str] = Query(None)):
    nsfw_allowed = get_profile_setting(profile_id, 'nsfw_enabled')
    content_ratings = ["safe", "suggestive"]
    if nsfw_allowed:
        content_ratings.extend(["erotica", "pornographic"])

    params = {
        "title": q,
        "limit": 24,
        "contentRating[]": content_ratings,
        "includes[]": ["cover_art"]
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MANGADEX_API_URL}/manga", params=params)
            resp.raise_for_status()
            data = resp.json()
            # Process data to add a simple cover_url
            for manga in data.get("data", []):
                manga["cover_url"] = _get_cover_url_from_manga(manga)
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/mangadex/list")
async def list_mangadex(
    order: str = Query("latestUploadedChapter", enum=["latestUploadedChapter", "followedCount", "createdAt", "updatedAt"]),
    limit: int = Query(20, ge=1, le=100),
    profile_id: Optional[str] = Query(None)
):
    nsfw_allowed = get_profile_setting(profile_id, 'nsfw_enabled')
    content_ratings = ["safe", "suggestive"]
    if nsfw_allowed:
        content_ratings.extend(["erotica", "pornographic"])

    params = {
        f"order[{order}]": "desc",
        "limit": limit,
        "contentRating[]": content_ratings,
        "includes[]": ["cover_art"]
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MANGADEX_API_URL}/manga", params=params)
            resp.raise_for_status()
            data = resp.json()
            # Process data to add a simple cover_url
            for manga in data.get("data", []):
                manga["cover_url"] = _get_cover_url_from_manga(manga)
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/mangadex/manga/{manga_id}")
async def get_mangadex_manga_details(manga_id: str):
    params = {"includes[]": ["cover_art", "author", "artist"]}
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MANGADEX_API_URL}/manga/{manga_id}", params=params)
            resp.raise_for_status()
            data = resp.json().get("data")
            if data:
                # Process data to add a simple image_url
                data["image_url"] = _get_cover_url_from_manga(data)
            return data
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/mangadex/cover/{manga_id}/{file_name}")
async def get_mangadex_cover(
    manga_id: str,
    file_name: str,
    size: int = Query(256, enum=[256, 512])
):
    """
    Fetches a manga cover image from MangaDex with the required referrer.
    """
    cover_url = f"https://uploads.mangadex.org/covers/{manga_id}/{file_name}.{size}.jpg"
    headers = {"Referer": "https://mangadex.org/"}
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(cover_url, headers=headers, timeout=20)
            resp.raise_for_status()
            
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            return StreamingResponse(io.BytesIO(resp.content), media_type=content_type)
        
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex cover API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching cover: {str(e)}")


@app.get("/mangadex/manga/{manga_id}/chapters")
async def get_mangadex_manga_chapters(
    manga_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    params = {
        "limit": limit,
        "offset": offset,
        "translatedLanguage[]": "en",
        "order[chapter]": "asc",
        "order[volume]": "asc",
        "includes[]": "scanlation_group",
        "contentRating[]": ["safe", "suggestive", "erotica", "pornographic"]
    }
    
    headers = {"Referer": "https://mangadex.org/"}
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{MANGADEX_API_URL}/manga/{manga_id}/feed", params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            
            return {
                "chapters": data.get("data", []),
                "total": data.get("total", 0)
            }
            
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/mangadex/manga/{manga_id}/all-chapters")
async def get_all_mangadex_manga_chapters(manga_id: str):
    """
    Fetches ALL chapter IDs for a given MangaDex manga by handling pagination.
    """
    all_chapter_ids = []
    limit = 100 
    offset = 0
    
    params = {
        "limit": limit,
        "translatedLanguage[]": "en",
        "order[chapter]": "asc",
        "order[volume]": "asc",
        "contentRating[]": ["safe", "suggestive", "erotica", "pornographic"]
    }
    headers = {"Referer": "https://mangadex.org/"}

    async with httpx.AsyncClient() as client:
        while True:
            try:
                params["offset"] = offset
                resp = await client.get(f"{MANGADEX_API_URL}/manga/{manga_id}/feed", params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                
                chapters_on_page = data.get("data", [])
                if not chapters_on_page:
                    break

                for chapter in chapters_on_page:
                    all_chapter_ids.append(chapter['id'])

                total = data.get("total", 0)
                offset += limit
                
                if offset >= total:
                    break
            
            except httpx.HTTPStatusError as e:
                raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error during pagination: {e.response.text}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"An unexpected error occurred during pagination: {str(e)}")
    
    return {"chapter_ids": all_chapter_ids}


@app.get("/mangadex/manga/{manga_id}/chapter-nav-details/{chapter_id}")
async def get_mangadex_chapter_nav_details(manga_id: str, chapter_id: str):
    all_chapters = []
    limit = 500
    offset = 0
    total = 0

    params = {
        "limit": limit,
        "offset": offset,
        "translatedLanguage[]": "en",
        "order[chapter]": "asc",
        "order[volume]": "asc",
        "includes[]": "scanlation_group",
        "contentRating[]": ["safe", "suggestive", "erotica", "pornographic"]
    }
    headers = {"Referer": "https://mangadex.org/"}

    async with httpx.AsyncClient() as client:
        try:
            # Loop to fetch all pages of chapters
            while True:
                params["offset"] = offset
                resp = await client.get(f"{MANGADEX_API_URL}/manga/{manga_id}/feed", params=params, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                
                chapters_on_page = data.get("data", [])
                all_chapters.extend(chapters_on_page)
                
                total = data.get("total", 0)
                offset += limit
                
                if offset >= total or not chapters_on_page:
                    break
            
            current_chapter_index = -1
            for i, chap in enumerate(all_chapters):
                if chap["id"] == chapter_id:
                    current_chapter_index = i
                    break
            
            if current_chapter_index == -1:
                raise HTTPException(status_code=404, detail="Chapter not found in manga feed.")

            current_chapter_details = all_chapters[current_chapter_index]
            
            next_chapter_id = None
            if current_chapter_index + 1 < len(all_chapters):
                next_chapter_id = all_chapters[current_chapter_index + 1]["id"]

            return {
                "current_chapter": current_chapter_details,
                "next_chapter_id": next_chapter_id,
                "total_chapters": total
            }

        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        except Exception as e:
            
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/mangadex/chapter/{chapter_id}")
async def get_mangadex_chapter_images(chapter_id: str):
    at_home_url = f"{MANGADEX_API_URL}/at-home/server/{chapter_id}"
    headers = {"Referer": "https://mangadex.org/"}

    async with httpx.AsyncClient() as client:
        try:
            print(f"Fetching images for chapter {chapter_id} from MangaDex... URL: {at_home_url}")
            
            server_resp = await client.get(at_home_url, headers=headers, timeout=20)
            server_resp.raise_for_status()
            server_data = server_resp.json()
            
            base_url = server_data.get("baseUrl")
            chapter_hash = server_data.get("chapter", {}).get("hash")
            page_filenames = server_data.get("chapter", {}).get("data", [])
            
            if not all([base_url, chapter_hash, page_filenames]):
                print(f"Error: Incomplete data from MangaDex for chapter {chapter_id}. Data: {server_data}")
                raise HTTPException(status_code=500, detail="Incomplete data from MangaDex server endpoint")

            sorted_filenames = sorted(page_filenames, key=natural_sort_key)
            parsed_url = urlparse(base_url)
            server_host = parsed_url.netloc
            
            image_urls = [f"/mangadex/proxy/{server_host}/data/{chapter_hash}/{filename}" for filename in sorted_filenames]
            
            return image_urls
        
        except httpx.HTTPStatusError as e:
            print(f"MangaDex API returned an error for {at_home_url}: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex API error: {e.response.text}")
        
        except json.JSONDecodeError as e:
            print(f"Failed to decode JSON from MangaDex for {at_home_url}. Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to parse response from MangaDex.")
            
        except Exception as e:
            print(f"An unexpected error occurred in get_mangadex_chapter_images for chapter {chapter_id}:")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/mangadex/proxy/{server_host}/data/{chapter_hash}/{filename:path}")
async def proxy_mangadex_image(server_host: str, chapter_hash: str, filename: str):
    """
    Proxies a chapter image from MangaDex with the required referrer.
    """
    # Basic validation to prevent open proxy abuse
    if not server_host.endswith("mangadex.network"):
        raise HTTPException(status_code=400, detail="Invalid server host for MangaDex proxy.")

    image_url = f"https://{server_host}/data/{chapter_hash}/{filename}"
    headers = {"Referer": "https://mangadex.org/"}
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(image_url, headers=headers, timeout=20)
            resp.raise_for_status()
            
            content_type = resp.headers.get("Content-Type", "image/jpeg")
            return StreamingResponse(io.BytesIO(resp.content), media_type=content_type)
        
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"MangaDex image proxy error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred while proxying image: {str(e)}")
        
# --- UPDATE THIS IN app.py ---

# --- UPDATE THIS IN app.py ---

# --- UPDATE THIS IN app.py ---

@app.get("/proxy")
async def generic_proxy(
    request: Request,
    url: str = Query(..., description="Target URL to fetch"),
    referer: Optional[str] = Query(None, description="Referer header to send")
):
    """
    Generic proxy that handles Referer, Range headers, and keeps the connection
    alive during streaming to prevent incomplete chunk errors.
    """
    if not url:
        raise HTTPException(status_code=400, detail="Missing URL parameter")

    # 1. Prepare Headers
    headers = {}
    if referer:
        headers["Referer"] = referer
    
    # Forward Range header (Critical for video seeking/playback)
    if "range" in request.headers:
        headers["Range"] = request.headers["range"]
    
    headers["User-Agent"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"

    # 2. Create Client (No 'async with' here, we manage lifecycle manually)
    client = httpx.AsyncClient(follow_redirects=True, verify=False)
    
    try:
        req = client.build_request("GET", url, headers=headers)
        r = await client.send(req, stream=True)
    except Exception as e:
        await client.aclose()
        print(f"Proxy Connection Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Handle Upstream Errors immediately
    if r.status_code >= 400:
        content = await r.aread()
        await r.aclose()
        await client.aclose()
        return Response(content=content, status_code=r.status_code)

    # 3. Filter Headers
    # We strip 'content-encoding' because aiter_bytes() will auto-decompress gzip
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection', 'host']
    response_headers = {
        k: v for k, v in r.headers.items() 
        if k.lower() not in excluded_headers
    }

    # 4. Define Stream Generator with Cleanup
    async def stream_content():
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        except Exception as e:
            print(f"Streaming Error: {e}")
        finally:
            # Critical: Close the client only AFTER streaming is done
            await r.aclose()
            await client.aclose()

    return StreamingResponse(
        stream_content(),
        status_code=r.status_code,
        headers=response_headers,
        media_type=r.headers.get("content-type")
    )


# --- Jikan Endpoints ---
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

@app.get("/chapters/{mal_id}")
async def get_manga_chapters(mal_id: int, profile_id: Optional[str] = Query(None, description="ID of the active user profile")):
    """
    Iterates through enabled manga modules to find a list of chapters.
    Requires NSFW setting to be enabled for NSFW modules.
    """
    current_profile = profiles.get(profile_id)
    nsfw_allowed = current_profile and current_profile.settings.nsfw_enabled
    module_prefs = current_profile.settings.module_preferences.get("Manga", []) if current_profile else []
    hentai_manga_prefs = current_profile.settings.module_preferences.get("Hentai (Manga)", []) if current_profile else []

    sorted_modules = []
    processed_module_ids = set()

    for module_id in module_prefs:
        if module_id in loaded_modules and not loaded_modules[module_id]["info"].get("nsfw", False):
            sorted_modules.append((module_id, loaded_modules[module_id]))
            processed_module_ids.add(module_id)
    
    for module_id in hentai_manga_prefs:
        if module_id in loaded_modules and loaded_modules[module_id]["info"].get("nsfw", False):
            sorted_modules.append((module_id, loaded_modules[module_id]))
            processed_module_ids.add(module_id)

    for module_id, module_data in sorted(loaded_modules.items()):
        if module_id not in processed_module_ids:
            sorted_modules.append((module_id, module_data))

    for module_id, module_data in sorted_modules:
        module_info = module_data.get("info", {})
        is_nsfw_module = module_info.get("nsfw", False)

        module_type = module_info.get("type")
        is_manga_reader = (isinstance(module_type, list) and "MANGA_READER" in module_type) or \
                          (isinstance(module_type, str) and module_type == "MANGA_READER")

        if module_states.get(module_id) and is_manga_reader:
            if is_nsfw_module and not nsfw_allowed:
                print(f"Skipping NSFW module {module_id} as NSFW content is not enabled for profile {profile_id}.")
                continue

            print(f"Attempting to fetch chapters from module: {module_id}")
            try:
                chapters_func = getattr(module_data["instance"], "get_chapters", None)
                if not chapters_func:
                    continue
                
                chapters = await chapters_func(mal_id)
                if chapters:
                    print(f"Success! Got {len(chapters)} chapters from {module_id}")
                    return {"chapters": chapters, "source_module": module_id}
            except Exception as e:
                print(f"Module {module_id} failed with an error: {e}")
                
    raise HTTPException(status_code=404, detail="Could not retrieve chapters from any enabled module or NSFW content is not enabled.")

async def _get_manga_images_from_modules(
    mal_id: int, 
    chapter_num: str, 
    profile_id: Optional[str]
) -> Optional[List[str]]:
    """
    A helper function to iterate through enabled manga modules to get all page images for a chapter.
    Requires NSFW setting to be enabled for NSFW modules.
    This is not an endpoint.
    """
    current_profile = profiles.get(profile_id)
    nsfw_allowed = current_profile and current_profile.settings.nsfw_enabled
    
    module_prefs = current_profile.settings.module_preferences.get("Manga", []) if current_profile else []
    hentai_manga_prefs = current_profile.settings.module_preferences.get("Hentai (Manga)", []) if current_profile else []

    sorted_modules = []
    processed_module_ids = set()

    for module_id in module_prefs:
        if module_id in loaded_modules and not loaded_modules[module_id]["info"].get("nsfw", False):
            sorted_modules.append((module_id, loaded_modules[module_id]))
            processed_module_ids.add(module_id)
    
    for module_id in hentai_manga_prefs:
        if module_id in loaded_modules and loaded_modules[module_id]["info"].get("nsfw", False):
            sorted_modules.append((module_id, loaded_modules[module_id]))
            processed_module_ids.add(module_id)

    for module_id, module_data in sorted(loaded_modules.items()):
        if module_id not in processed_module_ids:
            sorted_modules.append((module_id, module_data))

    for module_id, module_data in sorted_modules:
        module_info = module_data.get("info", {})
        is_nsfw_module = module_info.get("nsfw", False)

        module_type = module_info.get("type")
        is_manga_reader = (isinstance(module_type, list) and "MANGA_READER" in module_type) or \
                          (isinstance(module_type, str) and module_type == "MANGA_READER")

        if module_states.get(module_id) and is_manga_reader:
            if is_nsfw_module and not nsfw_allowed:
                print(f"Skipping NSFW module {module_id} as NSFW content is not enabled for profile {profile_id}.")
                continue

            print(f"Attempting to fetch chapter images from module: {module_id}")
            try:
                images_func = getattr(module_data["instance"], "get_chapter_images", None)
                if not images_func:
                    continue
                
                images = await images_func(mal_id, chapter_num)
                if images is not None:
                    print(f"Success! Got {len(images)} images from {module_id}")
                    return images
            except Exception as e:
                print(f"Module {module_id} failed with an error: {e}")
                traceback.print_exc()
    return None

@app.get("/retrieve/{mal_id}/{chapter_num}")
async def get_manga_chapter_images(
    mal_id: int, 
    chapter_num: str, 
    profile_id: Optional[str] = Query(None, description="ID of the active user profile"),
    ext: Optional[str] = Query(None, description="The ID of the extension to use")
):
    """
    Iterates through enabled manga modules or a specific extension to get all page images for a chapter.
    Requires NSFW setting to be enabled for NSFW modules/extensions.
    """
    # --- Handle Extension Request ---
    if ext:
        if ext in loaded_extensions:
            ext_data = loaded_extensions[ext]
            ext_info = ext_data.get("info", {})
            is_nsfw_ext = ext_info.get("nsfw", False)
            
            current_profile = profiles.get(profile_id)
            nsfw_allowed = current_profile and current_profile.settings.nsfw_enabled

            if is_nsfw_ext and not nsfw_allowed:
                raise HTTPException(status_code=403, detail=f"NSFW extension '{ext}' is disabled for this profile.")

            try:
                images_func = getattr(ext_data["instance"], "get_chapter_images", None)
                if images_func:
                    print(f"Attempting to fetch chapter images from extension: {ext}")
                    images = await images_func(mal_id, chapter_num)
                    if images is not None:
                        print(f"Success! Got {len(images)} images from extension {ext}")
                        return images
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Extension {ext} failed: {e}")
        else:
            raise HTTPException(status_code=404, detail=f"Extension '{ext}' not found.")

    # --- Handle Module Request (Legacy) ---
    images = await _get_manga_images_from_modules(mal_id, chapter_num, profile_id)
    if images is not None:
        return images
                
    raise HTTPException(status_code=404, detail="Could not retrieve chapter images from any enabled module or extension.")

@app.get("/player/templates/pdf_reader.html", response_class=HTMLResponse)
async def serve_pdf_reader():
    """
    Serves the offline PDF reader HTML page.
    """
    try:
        with open("templates/pdf_reader.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF Reader UI not found.")
    
@app.get("/map/file/animekai", response_class=JSONResponse)
async def serve_animekai_map():
    """
    Serves the AnimeKai map JSON file.
    """
    try:
        with open("templates/map.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            return JSONResponse(content=data)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="AnimeKai Map UI not found.")
        
@app.get("/read/{source}/{manga_id}/{chapter_id}", response_class=HTMLResponse)
async def read_manga_chapter_source(source: str, manga_id: str, chapter_id: str):
    """
    Serves the HTML reader interface for a given source (jikan or mangadex).
    """
    try:
        with open("templates/reader.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Reader UI not found.")

@app.get("/download-manga/site/{source}/{manga_id}/{chapter_id}", response_class=HTMLResponse)
async def download_manga_page(source: str, manga_id: str, chapter_id: str):
    """
    Serves the HTML download page.
    """
    try:
        with open("templates/download.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Download UI not found.")

from PIL import Image
import math

@app.get("/download-manga/direct/{source}/{manga_id}/{chapter_id}")
async def download_manga_chapter_as_pdf(source: str, manga_id: str, chapter_id: str, request: Request):
    """
    Generates and returns a PDF for a specific manga chapter from a given source.
    This function stitches all images into a single vertical strip and then slices it 
    into standard pages to correctly handle webtoon formats.
    """
    images = []
    manga_title = "Manga"
    chapter_num_str = chapter_id
    base_url = f"{request.url.scheme}://{request.url.netloc}"

    # 1. Fetch chapter images URLs
    if source == "mangadex":
        images = await get_mangadex_chapter_images(chapter_id)
        if images:
            try:
                details = await get_mangadex_manga_details(manga_id)
                manga_title = details.get("attributes", {}).get("title", {}).get("en", "MangaDex Manga")
            except Exception:
                manga_title = "MangaDex Manga"
    else:  # jikan/mal
        images = await _get_manga_images_from_modules(mal_id=int(manga_id), chapter_num=chapter_id, profile_id=None)
        if images:
            try:
                async with httpx.AsyncClient() as client:
                    jikan_url = f"https://api.jikan.moe/v4/manga/{manga_id}"
                    resp = await client.get(jikan_url)
                    resp.raise_for_status()
                    manga_title = resp.json().get("data", {}).get("title", "Manga")
            except Exception:
                manga_title = "Manga"

    if not images:
        raise HTTPException(status_code=404, detail="Could not retrieve chapter images.")

    # 2. Download images concurrently
    print(f"Downloading {len(images)} images concurrently...")
    async def fetch_image_content(client, image_url, base_url):
        try:
            full_image_url = image_url if image_url.startswith('http') else f"{base_url}{image_url}"
            response = await client.get(full_image_url, timeout=60)
            response.raise_for_status()
            return (image_url, response.content)
        except Exception as e:
            print(f"Failed to download image {image_url}: {e}")
            return (image_url, None)

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=10)) as client:
        tasks = [fetch_image_content(client, image_url, base_url) for image_url in images]
        image_results = await asyncio.gather(*tasks)

    # 3. Process and stitch images into a single composite
    print("All images downloaded. Stitching them together...")
    processed_images = []
    total_height = 0
    standard_width = 595  # A4 width in points

    for url, content in image_results:
        if content:
            try:
                with Image.open(io.BytesIO(content)) as img:
                    img = img.convert("RGB")
                    aspect_ratio = img.height / img.width
                    new_height = int(standard_width * aspect_ratio)
                    resized_img = img.resize((standard_width, new_height), Image.Resampling.LANCZOS)
                    processed_images.append(resized_img)
                    total_height += new_height
            except Exception as e:
                print(f"Could not process image {url}: {e}")

    if not processed_images:
        raise HTTPException(status_code=500, detail="No images could be processed.")

    composite_image = Image.new('RGB', (standard_width, total_height))
    current_y = 0
    for img in processed_images:
        composite_image.paste(img, (0, current_y))
        current_y += img.height

    # 4. Slice the composite image into PDF pages
    print("Slicing composite image into PDF pages...")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=False, margin=0)
    pdf.set_title(f"Chapter {chapter_num_str} - {manga_title}")

    page_height_pt = 842  # A4 height in points
    num_pages = math.ceil(total_height / page_height_pt)

    for i in range(num_pages):
        y_start = i * page_height_pt
        box = (0, y_start, standard_width, y_start + page_height_pt)
        page_image = composite_image.crop(box)

        with io.BytesIO() as page_buffer:
            page_image.save(page_buffer, format="PNG")
            page_buffer.seek(0)
            
            pdf.add_page()
            pdf.image(page_buffer, x=0, y=0, w=pdf.w)
            print(f"Added page {i+1}/{num_pages} to PDF.")

    # 5. Prepare and return PDF response
    pdf_output = pdf.output(dest='S')
    safe_title = "".join([c for c in manga_title if c.isalpha() or c.isdigit() or c.isspace()]).rstrip()
    filename = f"{safe_title} - Chapter {chapter_num_str}.pdf"
    
    return Response(
        content=bytes(pdf_output),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=\"{filename}\""}
    )


# --- Anime Endpoints ---

@app.get("/iframe-src")
async def get_iframe_source(
    mal_id: int = Query(..., description="MyAnimeList ID of the anime"),
    episode: int = Query(..., description="The episode number"),
    dub: bool = Query(False, description="Whether to fetch the dubbed version"),
    prefer_module: Optional[str] = Query(None, description="Exact module name to prefer (optional)")
):
    """
    Iterates through enabled modules to find an iframe source for the given anime episode.
    If 'prefer_module' is provided and matches an enabled module, only that module is used.
    """
    modules_to_try = []
    if prefer_module:
        # Only use the exact module if enabled
        mod = loaded_modules.get(prefer_module)
        if mod and module_states.get(prefer_module, False):
            modules_to_try = [(prefer_module, mod)]
        else:
            print(f"Preferred module '{prefer_module}' not found or not enabled. Falling back to normal order.")
            modules_to_try = [
                (module_id, module_data)
                for module_id, module_data in sorted(loaded_modules.items())
                if module_states.get(module_id, False)
            ]
    else:
        modules_to_try = [
            (module_id, module_data)
            for module_id, module_data in sorted(loaded_modules.items())
            if module_states.get(module_id, False)
        ]

    for module_id, module_data in modules_to_try:
        module_info = module_data.get("info", {})
        module_type = module_info.get("type")
        is_streamer = (isinstance(module_type, list) and "ANIME_STREAMER" in module_type) or \
                      (isinstance(module_type, str) and module_type == "ANIME_STREAMER")
        
        if not is_streamer:
            continue
        print(f"Attempting to fetch iframe source from module: {module_id}")
        try:
            source_func = getattr(module_data["instance"], "get_iframe_source", None)
            if not source_func:
                print(f"Module {module_id} does not have 'get_iframe_source' function.")
                continue
            
            iframe_src = await source_func(mal_id, episode, dub)

            if iframe_src:
                print(f"Success! Got source from {module_id}: {iframe_src}")
                return {"src": iframe_src, "source_module": module_id}
            else:
                print(f"Module {module_id} returned no source.")
        except Exception as e:
            print(f"Module {module_id} failed with an error: {e}")
            
    raise HTTPException(status_code=404, detail="Could not retrieve an iframe source from any enabled module.")


@app.get("/download")
async def get_download_link(
    mal_id: int = Query(..., description="MyAnimeList ID of the anime"),
    episode: int = Query(..., description="The episode number"),
    dub: bool = Query(False, description="Whether to fetch the dubbed version"),
    quality: str = Query("720p", description="The desired video quality (e.g., '1080p', '720p')")
):
    """
    Iterates through enabled modules to find a direct download link for the given anime episode.
    """
    for module_id, module_data in sorted(loaded_modules.items()):
        if module_states.get(module_id, False):
            module_info = module_data.get("info", {})
            module_type = module_info.get("type")
            is_downloader = (isinstance(module_type, list) and "ANIME_DOWNLOADER" in module_type) or \
                            (isinstance(module_type, str) and module_type == "ANIME_DOWNLOADER")
            
            if not is_downloader:
                continue

            print(f"Attempting to fetch download link from module: {module_id}")
            try:
                download_func = getattr(module_data["instance"], "get_download_link", None)
                if not download_func:
                    print(f"Module {module_id} does not have 'get_download_link' function.")
                    continue

                loop = asyncio.get_running_loop()
                download_link = await loop.run_in_executor(
                    None, download_func, mal_id, episode, dub, quality
                )

                if download_link:
                    print(f"Success! Got download link from {module_id}: {download_link}")
                    return {"download_link": download_link, "source_module": module_id}
                else:
                    print(f"Module {module_id} returned no download link.")
            except Exception as e:
                
                print(f"Module {module_id} failed with a download error: {e}")
                traceback.print_exc()
                
    raise HTTPException(status_code=404, detail="Could not retrieve a download link from any enabled module.")

@app.get("/map/mal/{mal_id}")
async def mal_to_kitsu(mal_id: int):
    ANIME_DB_URL = "https://raw.githubusercontent.com/Fribb/anime-lists/refs/heads/master/anime-offline-database-reduced.json"
    current_time = time.time()
    
    # 1. Check if we need to update the in-memory database
    # Update if empty OR older than 24 hours
    if MEMORY_CACHE["anime_db"] is None or (current_time - MEMORY_CACHE["anime_db_timestamp"] > 86400):
        print("Refreshing Anime Database for mappings...")
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(ANIME_DB_URL, timeout=30)
                resp.raise_for_status()
                MEMORY_CACHE["anime_db"] = resp.json()
                MEMORY_CACHE["anime_db_timestamp"] = current_time
        except Exception as e:
            # If update fails but we have old data, use old data
            if MEMORY_CACHE["anime_db"] is None:
                 raise HTTPException(status_code=500, detail=f"Failed to load anime mapping data: {str(e)}")
            print(f"Failed to refresh anime DB, using cached version: {e}")

    # 2. Look up the ID
    anime_list = MEMORY_CACHE["anime_db"]
    
    # Optimization: If the list is huge, converting to a dict once is faster, 
    # but for a snippet, simple iteration is safer to copy-paste.
    for anime in anime_list:
        if anime.get("mal_id") == mal_id:
            kitsu_id = anime.get("kitsu_id")
            if kitsu_id:
                return {"kitsu_id": kitsu_id}
            # If found but no kitsu_id, technically 404 for the mapping
            raise HTTPException(status_code=404, detail=f"No Kitsu ID found for MAL ID {mal_id}")
            
    raise HTTPException(status_code=404, detail=f"MAL ID {mal_id} not found in database")
    
ANILIST_API_URL = "https://graphql.anilist.co"
ANILIST_QUERY = """
query ($malId: Int) {
  Media(idMal: $malId, type: ANIME) {
    id
    bannerImage
    coverImage {
      extraLarge
      large
      medium
    }
  }
}
"""
@app.get("/anime/image")
async def get_anime_image(
    mal_id: int = Query(..., description="MyAnimeList anime ID"),
    cover: bool = Query(False, description="Return cover image instead of banner")
):
    payload = {
        "query": ANILIST_QUERY,
        "variables": {"malId": mal_id}
    }
    async with httpx.AsyncClient() as client:
        try:
            res = await client.post(ANILIST_API_URL, json=payload, timeout=10)
            res.raise_for_status()
            data = res.json()
            media = data.get("data", {}).get("Media")
            if not media:
                raise HTTPException(status_code=404, detail="Anime not found on AniList")
            
            if cover:
                image_url = (
                    media.get("coverImage", {}).get("extraLarge") or
                    media.get("coverImage", {}).get("large") or
                    media.get("coverImage", {}).get("medium")
                )
            else:
                image_url = media.get("bannerImage")
            
            if not image_url:
                raise HTTPException(status_code=404, detail="Image not available for this anime")

            img_response = await client.get(image_url, timeout=15)
            img_response.raise_for_status()
            
            content_type = img_response.headers.get("Content-Type", "image/jpeg")
            return StreamingResponse(io.BytesIO(img_response.content), media_type=content_type)
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"Error from upstream API: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


async def _fetch_jikan_with_retry(url: str, client: httpx.AsyncClient, retries=2, delay=1.0):
    # 1. Try to load from Cache first
    cached_data = load_cache(url)
    if cached_data:
        # print(f"Served from Cache: {url}") # Uncomment for debugging
        return cached_data

    # 2. Fetch if not in cache
    for attempt in range(retries):
        try:
            await asyncio.sleep(0.4) # Rate limit
            resp = await client.get(url, timeout=15)
            
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", delay))
                print(f"Rate limited. Waiting {retry_after}s...")
                await asyncio.sleep(retry_after)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            
            # 3. Save to Cache
            save_cache(url, data)
            return data
            
        except httpx.RequestError as e:
            if attempt == retries - 1:
                print(f"Final attempt failed for {url}: {e}")
                return None
            await asyncio.sleep(delay)
        except httpx.HTTPStatusError as e:
            # 404s should probably not be retried, just return None
            if e.response.status_code == 404:
                return None
            if attempt == retries - 1:
                return None
            await asyncio.sleep(delay)
            
    return None
async def _get_english_title(mal_id: int, client: httpx.AsyncClient):
    j = await _fetch_jikan_with_retry(f"https://api.jikan.moe/v4/anime/{mal_id}", client)
    if not j: return None
    data = j.get("data") or {}
    return data.get("title_english") or data.get("title")

@app.get("/anime/{mal_id}/seasons")
async def get_anime_seasons(mal_id: int):
    """
    Fetches a list of related anime seasons/entries for a given MAL ID.
    """
    async with httpx.AsyncClient() as client:
        relations_json = await _fetch_jikan_with_retry(f"https://api.jikan.moe/v4/anime/{mal_id}/relations", client)
        
        collected = {mal_id: {"mal_id": mal_id, "relation": "self"}}

        if relations_json and relations_json.get("data"):
            for rel in relations_json["data"]:
                rel_type = (rel.get("relation") or "").strip()
                if rel_type.lower() == "other":
                    continue
                for entry in rel.get("entry", []):
                    if entry.get("type") == "anime":
                        entry_id = entry.get("mal_id")
                        if not entry_id: continue
                        
                        if entry_id not in collected:
                            collected[entry_id] = {"mal_id": entry_id, "relation": rel_type}
                        else:
                            if rel_type not in collected[entry_id]["relation"]:
                                collected[entry_id]["relation"] += f", {rel_type}"

        title_tasks = [_get_english_title(mid, client) for mid in collected.keys()]
        titles = await asyncio.gather(*title_tasks)

        final_list = []
        for i, (mid, data) in enumerate(collected.items()):
            title = titles[i]
            if title:
                data["title"] = title
                final_list.append(data)

        if not any(item['mal_id'] == mal_id for item in final_list):
             raise HTTPException(status_code=404, detail="Could not fetch details for the requested anime.")

    return sorted(final_list, key=lambda x: x["mal_id"])

# --- Generic Extension Proxy ---
@app.api_route("/ext/{ext_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def proxy_to_extension(ext_name: str, path: str, request: Request):
    """
    Generic proxy for all extensions that run their own server.
    """
    ext_data = loaded_extensions.get(ext_name)
    if not ext_data or not ext_data.get("server_url"):
        raise HTTPException(status_code=404, detail=f"No running server found for extension '{ext_name}'.")

    target_url = ext_data["server_url"]
    target_url_full = f"{target_url}/{path}"
    print(f"Proxying request for '{ext_name}' to: {target_url_full}?{request.query_params}")

    async with httpx.AsyncClient() as client:
        try:
            headers = {k: v for k, v in request.headers.items() if k.lower() not in ["host", "content-length", "accept-encoding"]}
            
            response = await client.request(
                method=request.method,
                url=target_url_full,
                headers=headers,
                content=await request.body(),
                params=request.query_params,
                timeout=30.0
            )
            
            # The 'Content-Encoding' header is removed to prevent FastAPI/Uvicorn
            # from trying to decompress what is often an already-decompressed body.
            response_headers = {k:v for k,v in response.headers.items() if k.lower() != 'content-encoding'}

            return Response(content=response.content, status_code=response.status_code, headers=response_headers)

        except httpx.HTTPStatusError as e:
            print(f"Proxy HTTPStatusError: {e.response.status_code} - {e.response.text}")
            raise HTTPException(status_code=e.response.status_code, detail=f"Extension proxy error: {e.response.text}")
        except httpx.RequestError as e:
            print(f"Proxy RequestError: {e}")
            raise HTTPException(status_code=502, detail=f"Bad Gateway: Cannot connect to extension server for '{ext_name}'.")
        except Exception as e:
            print(f"Proxy Unexpected Error: {e}")
            raise HTTPException(status_code=500, detail=f"An unexpected error occurred during proxying: {e}")


# --- Static Files Hosting ---
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/", StaticFiles(directory="animex", html=True), name="static_site")
print("Static files mounted at /data and /")

# --- To make the server runnable directly ---
if __name__ == "__main__":
    animation_thread = threading.Thread(
        target=animate_loading, 
        args=(server_ready_event,), 
        daemon=True
    )
    animation_thread.start()
    uvicorn.run(app, host="0.0.0.0", port=7275, log_level="info")