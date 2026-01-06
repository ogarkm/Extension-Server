from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
import uvicorn
import sys
import os

try:
    # The server can now import the logic directly from the renamed logic.py file
    import extension as nsfw_logic
except ImportError as e:
    print(f"FATAL: Could not import the extension logic from 'logic.py'. Make sure the file exists and is in the same directory.")
    print(f"Import Error: {e}")
    sys.exit(1)

app = FastAPI()

@app.get("/hentai", response_class=FileResponse)
async def hentai_test_page(request: Request):
    """A simple test endpoint to confirm the server is running."""
    return FileResponse("src/hentai.html")

@app.get("/chapters", response_class=JSONResponse)
async def get_chapters_ext():
    """Proxies the call to the get_chapters_ext function in logic.py"""
    try:
        return JSONResponse(content=await nsfw_logic.get_chapters_ext())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chapter_images/{chapter_num}", response_class=JSONResponse)
async def get_chapter_images_ext(chapter_num: str):
    """Proxies the call to the get_chapter_images_ext function in logic.py"""
    try:
        return JSONResponse(content=await nsfw_logic.get_chapter_images_ext(chapter_num))
    except ValueError as e: # Catch specific ValueError from logic
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # This block is executed when running the start_command from the main app.
    uvicorn.run(app, host="0.0.0.0", port=7277)