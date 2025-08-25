import os
from pathlib import Path
from PIL import Image, ImageSequence

# исходная и целевая папка
SRC_DIR = Path("ideal_gifs")
DST_DIR = Path("ideal_gifs_looped")
DST_DIR.mkdir(parents=True, exist_ok=True)

# пройти по всем gif в подпапках
for gif_path in SRC_DIR.rglob("*.gif"):
    # путь сохранения в новой папке с сохранением структуры
    rel_path = gif_path.relative_to(SRC_DIR)
    out_path = DST_DIR / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # открыть гифку
    with Image.open(gif_path) as im:
        frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
        duration = im.info.get("duration", 100)  # задержка между кадрами (ms)

        # сохранить с loop=0 (бесконечный цикл)
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,  # вот здесь бесконечный цикл
            disposal=2,
        )

    print(f"✅ Saved looped GIF: {out_path}")
