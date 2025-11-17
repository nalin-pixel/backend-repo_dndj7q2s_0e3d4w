import os
import io
import uuid
import base64
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure generated assets directory exists and mount as static
GENERATED_DIR = os.path.join(os.getcwd(), "generated")
IMAGES_DIR = os.path.join(GENERATED_DIR, "images")
VIDEOS_DIR = os.path.join(GENERATED_DIR, "videos")
for d in [GENERATED_DIR, IMAGES_DIR, VIDEOS_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/generated", StaticFiles(directory=GENERATED_DIR), name="generated")


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        # Try to import database module
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"

            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    # Check environment variables
    import os as _os
    response["database_url"] = "✅ Set" if _os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if _os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


# ----------------------------
# Models
# ----------------------------
class GenerateImageRequest(BaseModel):
    prompt: str = Field(default="A beautiful abstract image")
    width: int = Field(default=768, ge=256, le=2048)
    height: int = Field(default=512, ge=256, le=2048)
    seed: Optional[int] = None
    style: str = Field(default="abstract")

class EditImageRequest(BaseModel):
    image_base64: str  # data URL or raw base64
    grayscale: bool = False
    blur: float = 0
    rotate: float = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    brightness: float = 1.0
    contrast: float = 1.0

class GenerateVideoRequest(BaseModel):
    prompt: str = Field(default="Hello World")
    width: int = Field(default=720, ge=320, le=1920)
    height: int = Field(default=1280, ge=320, le=1920)
    duration: float = Field(default=5.0, ge=1.0, le=30.0)
    fps: int = Field(default=24, ge=5, le=60)
    style: str = Field(default="neon")


# ----------------------------
# Utilities
# ----------------------------

def _save_pil_image(img, subdir: str = "images", ext: str = "png") -> str:
    filename = f"{uuid.uuid4().hex}.{ext}"
    dirpath = IMAGES_DIR if subdir == "images" else os.path.join(GENERATED_DIR, subdir)
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, filename)
    img.save(filepath)
    return f"/generated/{subdir}/{filename}"


def _decode_base64_image(data: str) -> bytes:
    # strip data URL prefix if present
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]
    return base64.b64decode(data)


# ----------------------------
# Image Generator (JSON)
# ----------------------------
@app.post("/api/generate-image")
async def generate_image(body: GenerateImageRequest):
    try:
        from PIL import Image, ImageDraw, ImageFont

        width = body.width
        height = body.height

        # Create gradient background based on seed
        if body.seed is not None:
            import random
            random.seed(int(body.seed))
        base = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(base)

        # Choose colors based on style
        palettes = {
            "sunset": [(255, 94, 98), (255, 195, 113), (255, 175, 123)],
            "ocean": [(56, 182, 255), (42, 123, 228), (15, 76, 129)],
            "forest": [(33, 147, 90), (123, 200, 164), (5, 102, 141)],
            "neon": [(0, 255, 164), (0, 75, 255), (255, 0, 229)],
            "abstract": [(59, 130, 246), (147, 51, 234), (236, 72, 153)],
        }
        colors = palettes.get(body.style.lower() if body.style else "abstract", palettes["abstract"])  # type: ignore

        # Vertical gradient
        for y in range(height):
            t = y / max(1, height - 1)
            if t < 0.5:
                c1, c2 = colors[0], colors[1]
                k = t / 0.5
            else:
                c1, c2 = colors[1], colors[2]
                k = (t - 0.5) / 0.5
            r = int(c1[0] * (1 - k) + c2[0] * k)
            g = int(c1[1] * (1 - k) + c2[1] * k)
            b = int(c1[2] * (1 - k) + c2[2] * k)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        # Overlay prompt text with shadow
        draw = ImageDraw.Draw(base)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(18, width // 20))
        except Exception:
            font = ImageFont.load_default()
        text = body.prompt[:120]
        tw, th = draw.textbbox((0, 0), text, font=font)[2:]
        x = (width - tw) // 2
        y = (height - th) // 2
        for dx, dy in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
            draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
        draw.text((x, y), text, font=font, fill=(255, 255, 255))

        url = _save_pil_image(base, subdir="images", ext="png")
        return {"url": url, "width": width, "height": height, "created_at": datetime.utcnow().isoformat()}
    except ModuleNotFoundError:
        raise HTTPException(status_code=500, detail="Pillow is not installed yet. Please retry in a moment.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ----------------------------
# Image Editor (JSON, base64 input)
# ----------------------------
@app.post("/api/edit-image")
async def edit_image(body: EditImageRequest):
    try:
        from PIL import Image, ImageFilter, ImageEnhance

        raw = _decode_base64_image(body.image_base64)
        img = Image.open(io.BytesIO(raw)).convert("RGBA")

        if body.grayscale:
            img = img.convert("L").convert("RGBA")
        if body.blur and float(body.blur) > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(body.blur)))
        if body.rotate and float(body.rotate) != 0:
            img = img.rotate(float(body.rotate), expand=True, resample=Image.BICUBIC)
        if body.flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if body.flip_vertical:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if body.brightness and float(body.brightness) != 1.0:
            img = ImageEnhance.Brightness(img).enhance(float(body.brightness))
        if body.contrast and float(body.contrast) != 1.0:
            img = ImageEnhance.Contrast(img).enhance(float(body.contrast))

        url = _save_pil_image(img.convert("RGBA"), subdir="images", ext="png")
        return {"url": url, "created_at": datetime.utcnow().isoformat()}
    except ModuleNotFoundError:
        raise HTTPException(status_code=500, detail="Pillow is not installed yet. Please retry in a moment.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {e}")


# ----------------------------
# Video Generator (JSON)
# ----------------------------
@app.post("/api/generate-video")
async def generate_video(body: GenerateVideoRequest):
    try:
        from moviepy.editor import ImageSequenceClip
        from PIL import Image, ImageDraw, ImageFont

        width = body.width
        height = body.height
        duration = body.duration
        fps = body.fps
        total_frames = int(duration * fps)

        palettes = {
            "sunset": [(255, 94, 98), (255, 195, 113), (255, 175, 123)],
            "ocean": [(56, 182, 255), (42, 123, 228), (15, 76, 129)],
            "forest": [(33, 147, 90), (123, 200, 164), (5, 102, 141)],
            "neon": [(0, 255, 164), (0, 75, 255), (255, 0, 229)],
            "abstract": [(59, 130, 246), (147, 51, 234), (236, 72, 153)],
        }
        colors = palettes.get((body.style or "abstract").lower(), palettes["abstract"])  # type: ignore

        frames = []
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=max(24, width // 12))
        except Exception:
            font = ImageFont.load_default()

        for i in range(total_frames):
            t = i / max(1, total_frames - 1)
            frame = Image.new("RGB", (width, height))
            draw = ImageDraw.Draw(frame)
            for y in range(height):
                gy = y / max(1, height - 1)
                if gy < 0.5:
                    c1, c2 = colors[0], colors[1]
                    k = gy / 0.5
                else:
                    c1, c2 = colors[1], colors[2]
                    k = (gy - 0.5) / 0.5
                r = int(c1[0] * (1 - k) + c2[0] * k)
                g = int(c1[1] * (1 - k) + c2[1] * k)
                b = int(c1[2] * (1 - k) + c2[2] * k)
                draw.line([(0, y), (width, y)], fill=(r, g, b))

            text = body.prompt[:60]
            tw, th = draw.textbbox((0, 0), text, font=font)[2:]
            x = int((width + tw) * (t) - tw)
            y = height // 2 - th // 2
            for dx, dy in [(-3, -3), (3, -3), (-3, 3), (3, 3)]:
                draw.text((x + dx, y + dy), text, font=font, fill=(0, 0, 0))
            draw.text((x, y), text, font=font, fill=(255, 255, 255))

            frames.append(frame)

        clip = ImageSequenceClip([frame for frame in frames], fps=fps)

        filename = f"{uuid.uuid4().hex}.mp4"
        out_path = os.path.join(VIDEOS_DIR, filename)
        clip.write_videofile(out_path, fps=fps, codec="libx264", audio=False, verbose=False, logger=None)

        url = f"/generated/videos/{filename}"
        return {"url": url, "width": width, "height": height, "duration": duration, "fps": fps}
    except ModuleNotFoundError as e:
        missing = "moviepy" if "moviepy" in str(e) else "Pillow"
        raise HTTPException(status_code=500, detail=f"{missing} is not installed yet. Please retry in a moment.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate video: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
