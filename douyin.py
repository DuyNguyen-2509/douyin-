#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Douyin Video Translator with Logo Blur
--------------------------------------
Tự động tải video Douyin, dịch và lồng tiếng sang nhiều ngôn ngữ,
đồng thời làm mờ logo (có thể tùy chỉnh vùng) trên video.
Yêu cầu: Python 3.10+, FFmpeg, Internet.
"""

import os
import sys
import subprocess
import tempfile
import shutil
import json
import asyncio
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Third-party imports ===
try:
    from moviepy.editor import VideoFileClip, vfx
except ImportError:
    print("[!] Missing moviepy. Run: pip install moviepy[all]")
    sys.exit(1)

try:
    import whisper
except ImportError:
    print("[!] Missing openai-whisper. Run: pip install openai-whisper")
    sys.exit(1)

try:
    from googletrans import Translator
except ImportError:
    print("[!] Missing googletrans==4.0.0-rc1. Run: pip install googletrans==4.0.0-rc1")
    sys.exit(1)

try:
    import edge_tts
except ImportError:
    print("[!] Missing edge-tts. Run: pip install edge-tts")
    sys.exit(1)

try:
    import pysrt
except ImportError:
    print("[!] Missing pysrt. Run: pip install pysrt")
    sys.exit(1)

try:
    import yt_dlp
except ImportError:
    print("[!] Missing yt-dlp. Run: pip install yt-dlp")
    sys.exit(1)

# === Configuration ===
@dataclass
class Config:
    input_source: str                     # URL Douyin hoặc file local
    target_lang: str = "vi"               # Mã ngôn ngữ đích (vi, en, fr, ...)
    source_lang: Optional[str] = None     # Tự động phát hiện nếu None
    output_dir: str = "output"
    output_filename: Optional[str] = None

    # Whisper
    whisper_model: str = "base"           # tiny/base/small/medium/large
    whisper_device: str = "cpu"           # cpu / cuda

    # TTS
    tts_voice: Optional[str] = None       # Ghi đè giọng đọc
    tts_gender: str = "female"            # male / female
    tts_rate: int = 0                     # Tốc độ (-50..+50)

    # Logo Blur
    blur_logo: bool = True                # Bật làm mờ logo
    logo_bbox: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2) – nếu None thì tự động phát hiện
    blur_radius: int = 25                 # Bán kính làm mờ

    # Subtitles
    burn_subtitles: bool = True           # Chèn phụ đề dịch
    subtitle_style: Dict[str, Any] = field(default_factory=lambda: {
        "font": "Arial",
        "fontsize": 24,
        "color": "white",
        "stroke_color": "black",
        "stroke_width": 2
    })

    # Background music
    preserve_music: bool = True
    music_ducking_db: float = 12.0

    # Performance
    max_workers: int = 4
    keep_temp: bool = False
    download_cookies: Optional[str] = None  # cookies.txt cho Douyin

# === Voice Map ===
VOICE_MAP = {
    "vi": {"female": "vi-VN-HoangMinhNeural", "male": "vi-VN-NamMinhNeural"},
    "en": {"female": "en-US-JennyNeural", "male": "en-US-GuyNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-JorgeNeural"},
    "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"},
    "ja": {"female": "ja-JP-NanamiNeural", "male": "ja-JP-KeitaNeural"},
    "zh": {"female": "zh-CN-XiaoxiaoNeural", "male": "zh-CN-YunxiNeural"},
    "ko": {"female": "ko-KR-SunHiNeural", "male": "ko-KR-InJoonNeural"},
    "pt": {"female": "pt-BR-FranciscaNeural", "male": "pt-BR-AntonioNeural"},
    "ru": {"female": "ru-RU-SvetlanaNeural", "male": "ru-RU-DmitryNeural"},
    "it": {"female": "it-IT-ElsaNeural", "male": "it-IT-DiegoNeural"},
    "ar": {"female": "ar-EG-SalmaNeural", "male": "ar-EG-ShakirNeural"},
    "hi": {"female": "hi-IN-SwaraNeural", "male": "hi-IN-MadhurNeural"},
}

# === Main Engine ===
class DouyinTranslatorWithLogoBlur:
    def __init__(self, config: Config):
        self.cfg = config
        self.temp_dir = tempfile.mkdtemp(prefix="douyin_logo_")
        self.translator = Translator()
        self.whisper_model = whisper.load_model(
            config.whisper_model,
            device=config.whisper_device
        )
        self.ffmpeg_path = shutil.which("ffmpeg")
        if not self.ffmpeg_path:
            raise RuntimeError("FFmpeg not found. Please install FFmpeg and add to PATH.")
        os.makedirs(config.output_dir, exist_ok=True)
        self._downloaded_path = None

    # ---------- Download Douyin ----------
    def _download_douyin(self, url: str) -> str:
        output_template = os.path.join(self.temp_dir, "douyin_%(id)s.%(ext)s")
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "merge_output_format": "mp4",
        }
        if self.cfg.download_cookies:
            ydl_opts["cookiefile"] = self.cfg.download_cookies

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if not os.path.exists(filename):
                for f in os.listdir(self.temp_dir):
                    if f.startswith("douyin_") and f.endswith(".mp4"):
                        filename = os.path.join(self.temp_dir, f)
                        break
        self._downloaded_path = filename
        return filename

    # ---------- Audio ----------
    def _extract_audio(self, video_path: str, audio_path: str) -> None:
        clip = VideoFileClip(video_path)
        if clip.audio is None:
            raise RuntimeError("Video không có âm thanh.")
        clip.audio.write_audiofile(audio_path, logger=None, verbose=False)
        clip.close()

    # ---------- Transcription ----------
    def _transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        result = self.whisper_model.transcribe(
            audio_path,
            task="transcribe",
            language=self.cfg.source_lang,
            word_timestamps=True,
            condition_on_previous_text=False
        )
        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"].strip()
            })
        return segments

    # ---------- Translation ----------
    def _translate_segments(self, segments: List[Dict[str, Any]], target_lang: str) -> List[Dict[str, Any]]:
        translated = []
        texts = [s["text"] for s in segments]
        try:
            translations = self.translator.translate(texts, dest=target_lang)
            if not isinstance(translations, list):
                translations = [translations]
        except Exception as e:
            print(f"[!] Lỗi dịch: {e}. Chuyển sang dịch từng đoạn.")
            translations = [self.translator.translate(t, dest=target_lang) for t in texts]

        for seg, trans in zip(segments, translations):
            translated.append({
                "start": seg["start"],
                "end": seg["end"],
                "original": seg["text"],
                "translated": trans.text if hasattr(trans, "text") else str(trans)
            })
        return translated

    # ---------- TTS ----------
    def _get_tts_voice(self) -> str:
        if self.cfg.tts_voice:
            return self.cfg.tts_voice
        lang = self.cfg.target_lang
        gender = self.cfg.tts_gender
        if lang in VOICE_MAP and gender in VOICE_MAP[lang]:
            return VOICE_MAP[lang][gender]
        return "en-US-JennyNeural"

    async def _tts_coro(self, text: str, output_path: str, voice: str):
        rate_str = f"+{self.cfg.tts_rate}%" if self.cfg.tts_rate >= 0 else f"{self.cfg.tts_rate}%"
        communicate = edge_tts.Communicate(text, voice, rate=rate_str)
        await communicate.save(output_path)

    def _generate_tts(self, text: str, output_path: str) -> None:
        voice = self._get_tts_voice()
        asyncio.run(self._tts_coro(text, output_path, voice))

    def _synthesize_dubbed_audio(self, segments: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[float, float]]]:
        audio_files = []
        timing = []

        def _gen_tts(i, seg):
            out_path = os.path.join(self.temp_dir, f"tts_{i:04d}.mp3")
            self._generate_tts(seg["translated"], out_path)
            return out_path, (seg["start"], seg["end"])

        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as executor:
            futures = {executor.submit(_gen_tts, i, seg): i for i, seg in enumerate(segments)}
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            results.sort(key=lambda x: x[1][0])
            audio_files = [r[0] for r in results]
            timing = [r[1] for r in results]

        concat_file = os.path.join(self.temp_dir, "dubbed_audio.mp3")
        if len(audio_files) == 1:
            shutil.copy(audio_files[0], concat_file)
        else:
            list_path = os.path.join(self.temp_dir, "concat_list.txt")
            with open(list_path, "w") as f:
                for af in audio_files:
                    f.write(f"file '{af}'\n")
            cmd = [
                self.ffmpeg_path, "-f", "concat", "-safe", "0",
                "-i", list_path, "-c", "copy", concat_file
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        return concat_file, timing

    # ---------- Logo Blur ----------
    def _detect_logo_bbox(self, video_path: str) -> Tuple[int, int, int, int]:
        """Tự động phát hiện vùng logo (mặc định góc trên trái, 10% chiều rộng và 15% chiều cao)."""
        probe_cmd = [
            self.ffmpeg_path, "-i", video_path,
            "-v", "quiet", "-print_format", "json", "-show_streams"
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        width, height = 1920, 1080
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                width = int(stream.get("width", 1920))
                height = int(stream.get("height", 1080))
                break
        # Giả định logo nằm ở góc trên trái, chiếm 10% chiều rộng và 15% chiều cao
        return (0, 0, int(width * 0.10), int(height * 0.15))

    def _blur_logo(self, video_path: str, output_path: str) -> None:
        if self.cfg.logo_bbox is None:
            bbox = self._detect_logo_bbox(video_path)
        else:
            bbox = self.cfg.logo_bbox
        x1, y1, x2, y2 = bbox
        # Dùng ffmpeg delogo để làm mờ vùng
        cmd = [
            self.ffmpeg_path, "-i", video_path,
            "-vf", f"delogo=x={x1}:y={y1}:w={x2-x1}:h={y2-y1}:band={self.cfg.blur_radius}",
            "-c:a", "copy", output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    # ---------- Burn Subtitles ----------
    def _burn_subtitles(self, video_path: str, segments: List[Dict[str, Any]], output_path: str) -> None:
        srt_path = os.path.join(self.temp_dir, "subtitles.srt")
        subs = []
        for i, seg in enumerate(segments, 1):
            start = seg["start"]
            end = seg["end"]
            text = seg["translated"]
            subs.append(f"{i}\n{self._format_time(start)} --> {self._format_time(end)}\n{text}\n")
        with open(srt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(subs))

        style = self.cfg.subtitle_style
        cmd = [
            self.ffmpeg_path, "-i", video_path,
            "-vf", f"subtitles={srt_path}:force_style='FontName={style['font']},FontSize={style['fontsize']},PrimaryColour=&H{self._hex_color(style['color'])},OutlineColour=&H{self._hex_color(style['stroke_color'])},Outline={style['stroke_width']}'",
            "-c:a", "copy", output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    @staticmethod
    def _format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _hex_color(color_name: str) -> str:
        mapping = {
            "white": "FFFFFF", "black": "000000",
            "red": "FF0000", "green": "00FF00",
            "blue": "0000FF", "yellow": "FFFF00",
            "cyan": "00FFFF", "magenta": "FF00FF",
        }
        return mapping.get(color_name.lower(), "FFFFFF")

    # ---------- Audio Mix ----------
    def _duck_music(self, video_path: str, dubbed_audio: str, output_path: str) -> None:
        orig_audio = os.path.join(self.temp_dir, "orig_audio.mp3")
        cmd_extract = [self.ffmpeg_path, "-i", video_path, "-q:a", "0", "-map", "a", orig_audio]
        subprocess.run(cmd_extract, check=True, capture_output=True)
        duck_cmd = [
            self.ffmpeg_path, "-i", orig_audio, "-i", dubbed_audio,
            "-filter_complex",
            f"[0:a]volume=volume=-{self.cfg.music_ducking_db}dB[a0];[1:a][a0]amix=inputs=2:duration=longest[a]",
            "-map", "[a]", output_path
        ]
        subprocess.run(duck_cmd, check=True, capture_output=True)

    # ---------- Main Pipeline ----------
    def run(self) -> str:
        print("[*] Bắt đầu xử lý video Douyin...")

        # Bước 0: Tải video nếu là URL
        if re.match(r'^https?://', self.cfg.input_source):
            print("[*] Đang tải video từ Douyin...")
            video_path = self._download_douyin(self.cfg.input_source)
        else:
            video_path = self.cfg.input_source

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy video: {video_path}")

        base_name = Path(video_path).stem
        out_name = self.cfg.output_filename or f"{base_name}_dubbed_{self.cfg.target_lang}.mp4"
        output_video = os.path.join(self.cfg.output_dir, out_name)

        # Bước 1: Trích xuất âm thanh
        audio_path = os.path.join(self.temp_dir, "audio.wav")
        print("[*] Trích xuất âm thanh...")
        self._extract_audio(video_path, audio_path)

        # Bước 2: Phát hiện ngôn ngữ nguồn
        if not self.cfg.source_lang:
            result = self.whisper_model.transcribe(audio_path, task="transcribe")
            self.cfg.source_lang = result.get("language", "en")
            print(f"[*] Ngôn ngữ nguồn: {self.cfg.source_lang}")

        # Bước 3: Phiên âm
        print("[*] Phiên âm...")
        segments = self._transcribe(audio_path)
        if not segments:
            raise RuntimeError("Không phát hiện giọng nói.")

        # Bước 4: Dịch
        print(f"[*] Dịch sang {self.cfg.target_lang}...")
        translated_segments = self._translate_segments(segments, self.cfg.target_lang)

        # Bước 5: Tạo audio lồng tiếng
        print("[*] Tạo giọng đọc TTS...")
        dubbed_audio_path, timing = self._synthesize_dubbed_audio(translated_segments)

        # Bước 6: Thay thế audio
        print("[*] Ghép audio mới...")
        temp_video = os.path.join(self.temp_dir, "no_audio.mp4")
        cmd_remove = [self.ffmpeg_path, "-i", video_path, "-an", "-c", "copy", temp_video]
        subprocess.run(cmd_remove, check=True, capture_output=True)

        temp_with_audio = os.path.join(self.temp_dir, "with_audio.mp4")
        if self.cfg.preserve_music:
            mixed_audio = os.path.join(self.temp_dir, "mixed_audio.mp3")
            self._duck_music(video_path, dubbed_audio_path, mixed_audio)
            cmd_add = [
                self.ffmpeg_path, "-i", temp_video, "-i", mixed_audio,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", temp_with_audio
            ]
        else:
            cmd_add = [
                self.ffmpeg_path, "-i", temp_video, "-i", dubbed_audio_path,
                "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0",
                "-shortest", temp_with_audio
            ]
        subprocess.run(cmd_add, check=True, capture_output=True)

        current_video = temp_with_audio

        # Bước 7: Làm mờ logo (nếu bật)
        if self.cfg.blur_logo:
            print("[*] Làm mờ logo...")
            blurred = os.path.join(self.temp_dir, "blurred_logo.mp4")
            self._blur_logo(current_video, blurred)
            current_video = blurred

        # Bước 8: Chèn phụ đề (nếu bật)
        if self.cfg.burn_subtitles:
            print("[*] Chèn phụ đề...")
            final_video = output_video
            self._burn_subtitles(current_video, translated_segments, final_video)
        else:
            final_video = output_video
            shutil.copy(current_video, final_video)

        # Dọn dẹp
        if not self.cfg.keep_temp:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        else:
            print(f"[*] Giữ file tạm tại: {self.temp_dir}")

        print(f"[+] Hoàn thành! Video đầu ra: {final_video}")
        return final_video

# === CLI ===
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Douyin Translator with Logo Blur – Tải, dịch, lồng tiếng và làm mờ logo video Douyin."
    )
    parser.add_argument("input", help="Đường dẫn video local hoặc URL Douyin")
    parser.add_argument("-tl", "--target-lang", default="vi", help="Mã ngôn ngữ đích (vi, en, fr, ...)")
    parser.add_argument("-sl", "--source-lang", help="Mã ngôn ngữ nguồn (tự động phát hiện nếu không set)")
    parser.add_argument("-o", "--output-dir", default="output", help="Thư mục lưu kết quả")
    parser.add_argument("-on", "--output-name", help="Tên file đầu ra (tùy chọn)")
    parser.add_argument("-wm", "--whisper-model", default="base", choices=["tiny","base","small","medium","large"],
                        help="Mô hình Whisper")
    parser.add_argument("--device", default="cpu", choices=["cpu","cuda"], help="Thiết bị chạy Whisper")
    parser.add_argument("--tts-voice", help="Ghi đè giọng đọc Edge TTS")
    parser.add_argument("--tts-gender", default="female", choices=["male","female"], help="Giới tính giọng đọc")
    parser.add_argument("--tts-rate", type=int, default=0, help="Tốc độ đọc (-50 đến +50)")
    parser.add_argument("--no-blur", action="store_true", help="Tắt làm mờ logo")
    parser.add_argument("--logo-bbox", help="Vùng logo: x1,y1,x2,y2 (mặc định tự động phát hiện)")
    parser.add_argument("--blur-radius", type=int, default=25, help="Bán kính làm mờ logo")
    parser.add_argument("--no-burn", action="store_true", help="Tắt chèn phụ đề")
    parser.add_argument("--no-music", action="store_true", help="Tắt giữ nhạc nền")
    parser.add_argument("--music-ducking", type=float, default=12.0, help="Mức giảm âm lượng nhạc nền (dB)")
    parser.add_argument("--workers", type=int, default=4, help="Số luồng xử lý TTS song song")
    parser.add_argument("--keep-temp", action="store_true", help="Giữ file tạm")
    parser.add_argument("--cookies", help="File cookies.txt để tải Douyin (nếu cần đăng nhập)")

    args = parser.parse_args()

    bbox = None
    if args.logo_bbox:
        bbox = tuple(map(int, args.logo_bbox.split(",")))
        if len(bbox) != 4:
            print("[!] Logo bbox phải có 4 số nguyên: x1,y1,x2,y2")
            sys.exit(1)

    config = Config(
        input_source=args.input,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        output_dir=args.output_dir,
        output_filename=args.output_name,
        whisper_model=args.whisper_model,
        whisper_device=args.device,
        tts_voice=args.tts_voice,
        tts_gender=args.tts_gender,
        tts_rate=args.tts_rate,
        blur_logo=not args.no_blur,
        logo_bbox=bbox,
        blur_radius=args.blur_radius,
        burn_subtitles=not args.no_burn,
        preserve_music=not args.no_music,
        music_ducking_db=args.music_ducking,
        max_workers=args.workers,
        keep_temp=args.keep_temp,
        download_cookies=args.cookies
    )

    engine = DouyinTranslatorWithLogoBlur(config)
    engine.run()

if __name__ == "__main__":
    main()