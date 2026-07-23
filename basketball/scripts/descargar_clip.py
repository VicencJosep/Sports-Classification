import subprocess

url = "https://www.youtube.com/watch?v=CXLM08fZO5o"

subprocess.run([
    "yt-dlp",
    "-f", "mp4",
    "--download-sections", "*0:00-0:04",
    "--force-keyframes-at-cuts",  # Garantiza cortes precisos
    "-o", "ja_morant-block.mp4",
    url
])