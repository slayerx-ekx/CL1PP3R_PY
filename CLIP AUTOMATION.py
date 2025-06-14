import os
import sys
import json
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QSpinBox, 
    QProgressBar, QMessageBox, QCheckBox, QFrame, QGroupBox, QTextEdit,
    QScrollArea, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QPalette, QColor, QIcon, QPixmap
from moviepy.editor import VideoFileClip
import whisper
import yt_dlp
import ffmpeg
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import spacy

os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"

class ContentAnalyzer:
    def __init__(self):
        # Model untuk analisis sentimen dan pentingnya kalimat
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentiment = pipeline("sentiment-analysis")
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def analyze_transcript(self, transcript):
        """Menganalisis transkrip dan menemukan bagian-bagian menarik"""
        doc = self.nlp(transcript)
        sentences = [sent.text for sent in doc.sents]
        
        scored_sentences = []
        for sent in sentences:
            sentiment = self.sentiment(sent)[0]
            importance = len(sent.split())  
            # Hitung skor
            score = 0
            if sentiment['label'] == 'POSITIVE':
                score += sentiment['score'] * 10
            else:
                score += (1 - sentiment['score']) * 5
            
            score += min(importance * 0.1, 3)            
            scored_sentences.append({
                'text': sent,
                'score': round(score, 1),
                'start_time': 0, 
                'end_time': 0    
            })
        
        # Urutkan berdasarkan skor
        scored_sentences.sort(key=lambda x: x['score'], reverse=True)
        
        return scored_sentences[:10]
    
    def generate_title(self, sentence):
        """Membuat judul menarik dari kalimat"""
        summary = self.summarizer(sentence, max_length=20, min_length=5, do_sample=False)
        return summary[0]['summary_text'].strip()

class ModernButton(QPushButton):
    def __init__(self, text, primary=False):
        super().__init__(text)
        self.primary = primary
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.update_style()
    
    def update_style(self):
        if self.primary:
            self.setStyleSheet("""
                QPushButton {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #4CAF50, stop:1 #45a049);
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 20px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #5CBF60, stop:1 #55b059);
                }
                QPushButton:pressed {
                    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 #3d8b40, stop:1 #359739);
                }
                QPushButton:disabled {
                    background: #cccccc;
                    color: #666666;
                }
            """)
        else:
            self.setStyleSheet("""
                QPushButton {
                    background: #f8f9fa;
                    color: #495057;
                    border: 2px solid #dee2e6;
                    border-radius: 8px;
                    padding: 8px 16px;
                }
                QPushButton:hover {
                    background: #e9ecef;
                    border-color: #adb5bd;
                }
                QPushButton:pressed {
                    background: #dee2e6;
                }
            """)

class ModernLineEdit(QLineEdit):
    def __init__(self, placeholder=""):
        super().__init__()
        self.setPlaceholderText(placeholder)
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QLineEdit {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 8px 12px;
                background: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #4CAF50;
                outline: none;
            }
            QLineEdit:hover {
                border-color: #adb5bd;
            }
        """)

class ModernComboBox(QComboBox):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QComboBox {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 8px 12px;
                background: white;
                min-width: 100px;
            }
            QComboBox:hover {
                border-color: #adb5bd;
            }
            QComboBox:focus {
                border-color: #4CAF50;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #666;
                margin-right: 10px;
            }
        """)

class ModernSpinBox(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(40)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QSpinBox {
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 8px 12px;
                background: white;
            }
            QSpinBox:hover {
                border-color: #adb5bd;
            }
            QSpinBox:focus {
                border-color: #4CAF50;
            }
        """)

class ModernCheckBox(QCheckBox):
    def __init__(self, text):
        super().__init__(text)
        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet("""
            QCheckBox {
                spacing: 10px;
                color: #495057;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #dee2e6;
                background: white;
            }
            QCheckBox::indicator:hover {
                border-color: #4CAF50;
            }
            QCheckBox::indicator:checked {
                background: #4CAF50;
                border-color: #4CAF50;
            }
            QCheckBox::indicator:checked:hover {
                background: #45a049;
            }
        """)

class ModernProgressBar(QProgressBar):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(12)
        self.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 6px;
                background: #e9ecef;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:1 #66BB6A);
                border-radius: 6px;
            }
        """)

class JsonProcessingThread(QThread):
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, worker, json_path):
        super().__init__()
        self.worker = worker
        self.json_path = json_path
    
    def run(self):
        success, message = self.worker.process_json_clips(self.json_path)
        self.finished_signal.emit(success, message)

class Worker(QThread):
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    log_updated = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.video_path = ""
        self.output_dir = ""
        self.clip_duration = 30
        self.add_subtitles = True
        self.whisper_model = "base"
        self.vertical_format = True
        self.is_youtube_url = False

    def run(self):
        try:
            # Handle YouTube URL
            if self.is_youtube_url:
                self.status_updated.emit("Downloading video from YouTube...")
                self.log_updated.emit("🔄 Starting YouTube download...")
                video_path = self.download_youtube_video()
                if not video_path:
                    self.finished.emit(False, "Failed to download YouTube video")
                    return
                self.video_path = video_path

            # Recommeded
            if hasattr(self, 'run_auto_recommend') and self.run_auto_recommend:
                clips = self.find_interesting_clips(self.video_path)
                if not clips:
                    self.finished.emit(False, "No interesting moments found")
                    return

                self.status_updated.emit(f"Found {len(clips)} interesting moments")
                self.log_updated.emit(f"🎯 Found {len(clips)} interesting moments to clip")

                os.makedirs(self.output_dir, exist_ok=True)

                video = VideoFileClip(self.video_path)
                total_clips = len(clips)
                processed_clips = 0

                for clip_info in clips:
                    start = clip_info['start']
                    end = clip_info['end']

                    self.status_updated.emit(f"Creating clip {processed_clips + 1}/{total_clips}")
                    self.log_updated.emit(f"✂️ Cutting interesting clip {start}s-{end}s...")

                    clip = video.subclip(start, end)

                    if self.vertical_format:
                        clip = self.convert_to_vertical(clip)

                    clip_filename = f"{clip_info['index']:02d}_{clip_info['score']:.1f}_{clip_info['title'][:30].replace(' ', '_')}.mp4"
                    clip_path = os.path.join(self.output_dir, clip_filename)

                    clip.write_videofile(
                        clip_path,
                        codec="libx264",
                        audio_codec="aac",
                        temp_audiofile="temp-audio.m4a",
                        remove_temp=True,
                        verbose=False,
                        logger=None
                    )

                    if self.add_subtitles:
                        self.status_updated.emit(f"Adding subtitles to clip {processed_clips + 1}/{total_clips}")
                        self.log_updated.emit(f"📝 Generating subtitles for {clip_filename}...")
                        subtitle_path = self.generate_subtitle(clip_path)
                        if subtitle_path:
                            final_path = clip_path.replace(".mp4", "_with_subs.mp4")
                            self.add_subtitle_to_video(clip_path, subtitle_path, final_path)
                            os.remove(clip_path)
                            os.rename(final_path, clip_path)

                    processed_clips += 1
                    progress = int(100 * processed_clips / total_clips)
                    self.progress_updated.emit(progress)
                    self.log_updated.emit(f"✅ Completed clip {processed_clips}/{total_clips}")

                video.close()
                self.log_updated.emit(f"🎉 Successfully processed {processed_clips} recommended clips!")
                self.finished.emit(True, f"Successfully created {processed_clips} recommended clips in {self.output_dir}")

            else:
                # Mode normal
                self.status_updated.emit("Processing video...")
                self.log_updated.emit(f"📹 Processing video: {Path(self.video_path).name}")

                os.makedirs(self.output_dir, exist_ok=True)

                video = VideoFileClip(self.video_path)
                duration = video.duration
                total_clips = int(duration // self.clip_duration) + (1 if duration % self.clip_duration > 0 else 0)
                processed_clips = 0

                self.log_updated.emit(f"📊 Video duration: {duration:.1f}s, will create {total_clips} clips")

                for i in range(0, int(duration), self.clip_duration):
                    start = i
                    end = min(i + self.clip_duration, duration)

                    self.status_updated.emit(f"Creating clip {processed_clips + 1}/{total_clips}")
                    self.log_updated.emit(f"✂️ Cutting clip {start}s-{end}s...")

                    clip = video.subclip(start, end)

                    if self.vertical_format:
                        clip = self.convert_to_vertical(clip)

                    clip_filename = f"clip_{int(start):03d}_{int(end):03d}.mp4"
                    clip_path = os.path.join(self.output_dir, clip_filename)

                    clip.write_videofile(
                        clip_path,
                        codec="libx264",
                        audio_codec="aac",
                        temp_audiofile="temp-audio.m4a",
                        remove_temp=True,
                        verbose=False,
                        logger=None
                    )

                    if self.add_subtitles:
                        self.status_updated.emit(f"Adding subtitles to clip {processed_clips + 1}/{total_clips}")
                        self.log_updated.emit(f"📝 Generating subtitles for {clip_filename}...")
                        subtitle_path = self.generate_subtitle(clip_path)
                        if subtitle_path:
                            final_path = clip_path.replace(".mp4", "_with_subs.mp4")
                            self.add_subtitle_to_video(clip_path, subtitle_path, final_path)
                            os.remove(clip_path)
                            os.rename(final_path, clip_path)

                    processed_clips += 1
                    progress = int(100 * processed_clips / total_clips)
                    self.progress_updated.emit(progress)
                    self.log_updated.emit(f"✅ Completed clip {processed_clips}/{total_clips}")

                video.close()
                self.log_updated.emit(f"🎉 Successfully processed {processed_clips} clips!")
                self.finished.emit(True, f"Successfully created {processed_clips} clips in {self.output_dir}")

        except Exception as e:
            self.log_updated.emit(f"❌ Error: {str(e)}")
            self.finished.emit(False, f"Error: {str(e)}")


    def download_youtube_video(self):
        """Download video from YouTube using yt-dlp"""
        try:
            # Create downloads directory
            download_dir = os.path.join(self.output_dir, "downloads")
            os.makedirs(download_dir, exist_ok=True)
            
            ydl_opts = {
                'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]',  # Format lebih fleksibel
                'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
                'quiet': True,
                'no_warnings': True,
                'ignoreerrors': True,
                'merge_output_format': 'mp4',
                'postprocessors': [{
                    'key': 'FFmpegVideoConvertor',
                    'preferedformat': 'mp4',
                }]
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.video_path, download=False)
                title = info.get('title', 'video')
                
                self.log_updated.emit(f"📺 Found video: {title}")
                ydl.download([self.video_path])
                
                # Find the downloaded file
                for file in os.listdir(download_dir):
                    if file.startswith(title.replace('/', '_')[:50]):  # Handle long titles
                        return os.path.join(download_dir, file)
                
                # Fallback: get the newest file
                files = [os.path.join(download_dir, f) for f in os.listdir(download_dir)]
                if files:
                    return max(files, key=os.path.getctime)
                
        except Exception as e:
            self.log_updated.emit(f"❌ YouTube download error: {str(e)}")
            return None

    def convert_to_vertical(self, clip):
        """Convert video to vertical 9:16 format"""
        w, h = clip.size
        target_ratio = 9 / 16
        current_ratio = w / h
        
        if current_ratio > target_ratio:
            # Video is too wide, crop width
            new_w = int(h * target_ratio)
            x_center = w / 2
            x1 = x_center - (new_w / 2)
            x2 = x_center + (new_w / 2)
            cropped = clip.crop(x1=x1, y1=0, x2=x2, y2=h)
        else:
            # Video is too tall, crop height
            new_h = int(w / target_ratio)
            y_center = h / 2
            y1 = y_center - (new_h / 2)
            y2 = y_center + (new_h / 2)
            cropped = clip.crop(x1=0, y1=y1, x2=w, y2=y2)
        
        # Resize to 1080x1920 (Instagram/YouTube Shorts format)
        return cropped.resize(newsize=(1080, 1920))
    
    def find_interesting_clips(self, video_path):
            """Temukan bagian menarik dari video secara otomatis"""
            try:
                self.status_updated.emit("Analyzing video content...")
                self.log_updated.emit("🔍 Analyzing video for interesting moments...")
                
                # Transkripsi video
                model = whisper.load_model(self.whisper_model)
                result = model.transcribe(video_path, language="id")
                
                # Gabungkan semua teks
                full_text = " ".join([seg['text'] for seg in result['segments']])
                
                # Analisis konten
                analyzer = ContentAnalyzer()
                interesting_parts = analyzer.analyze_transcript(full_text)
                
                # Map waktu dari transkrip
                for part in interesting_parts:
                    for seg in result['segments']:
                        if part['text'] in seg['text']:
                            part['start_time'] = seg['start']
                            part['end_time'] = seg['end']
                            break
                
                # Buat klip berdasarkan bagian menarik
                clips = []
                for i, part in enumerate(interesting_parts):
                    if part['start_time'] == 0 or part['end_time'] == 0:
                        continue
                        
                    duration = part['end_time'] - part['start_time']
                    if duration > 60:  # Batasi maksimal 60 detik
                        part['end_time'] = part['start_time'] + 60
                        
                    clips.append({
                        'start': part['start_time'],
                        'end': part['end_time'],
                        'score': part['score'],
                        'title': analyzer.generate_title(part['text']),
                        'index': i+1
                    })
                
                return clips
            
            except Exception as e:
                self.log_updated.emit(f"❌ Error analyzing content: {str(e)}")
                return []

    def generate_subtitle(self, video_path):
        """Generate .ass subtitle using Whisper with styling and fade effects"""
        try:
            model = whisper.load_model(self.whisper_model)
            result = model.transcribe(video_path, language="id")

            ass_path = video_path.replace(".mp4", ".ass")

            with open(ass_path, "w", encoding="utf-8") as f:
                # Header & Styles
                f.write("[Script Info]\n")
                f.write("Title: YouTube Shorts Subtitle\n")
                f.write("ScriptType: v4.00+\n")
                f.write("PlayResX: 1080\n")
                f.write("PlayResY: 1920\n")
                f.write("ScaledBorderAndShadow: yes\n\n")

                f.write("[V4+ Styles]\n")
                f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
                preset_name = getattr(self, 'subtitle_style', 'Simple')
                presets = self.get_style_presets()
                preset = presets.get(preset_name, presets['Simple'])  # fallback

                style_line = (
                    f"Style: Default,{preset['font']},{preset['font_size']},{preset['primary_color']},"
                    f"&H000000FF,{preset['outline_color']},{preset['back_color']},{preset['bold']},"
                    f"{preset['italic']},{preset['border']},{preset['outline']},{preset['shadow']},"
                    f"{preset['alignment']},{preset['margin_l']},{preset['margin_r']},{preset['margin_v']},1"
                )

                f.write(style_line + "\n\n")

                # Events (dialogues)
                f.write("[Events]\n")
                f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

                for seg in result["segments"]:
                    start = self.format_ass_time(seg['start'])
                    end = self.format_ass_time(seg['end'])
                    text = seg['text'].strip().replace("\n", " ").replace(",", "\\,")
                    effect = preset.get("effect", "{\\fad(300,300)}")
                    f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{effect}{text}\n")

            return ass_path

        except Exception as e:
            self.log_updated.emit(f"⚠️ Subtitle generation failed: {str(e)}")
            return None


    def format_ass_time(self, seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:d}:{m:02d}:{s:05.2f}".replace(".", ".")

    def format_time(self, seconds):
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

    def add_subtitle_to_video(self, video_path, subtitle_path, output_path):
        """Add subtitles to video using ffmpeg"""
        try:
            # Escape Windows path (replace \ with / and escape special characters)
            subtitle_path_escaped = subtitle_path.replace("\\", "/").replace(":", "\\:")
            
            (
                ffmpeg
                .input(video_path)
                .output(output_path, vf=f"ass='{subtitle_path_escaped}'")
                .overwrite_output()
                .run(quiet=True)
            )
            os.remove(subtitle_path)
        except Exception as e:
            self.log_updated.emit(f"⚠️ FFmpeg subtitle embedding failed: {str(e)}")
            raise Exception(f"Failed to add subtitles: {str(e)}")

    def parse_time(self, time_str):
        """Parse SRT time format to seconds"""
        time_str = time_str.replace(',', '.')
        h, m, s = time_str.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)
    
    def get_style_presets(self):
        """Return dictionary of subtitle style presets"""
        return {
            "Karaoke": {
                "font": "Arial",
                "font_size": 48,
                "primary_color": "&H00FFFFFF",
                "outline_color": "&H00000000",
                "back_color": "&H80000000",
                "bold": -1,
                "italic": 0,
                "border": 1,
                "outline": 2,
                "shadow": 1,
                "alignment": 2,
                "margin_l": 30,
                "margin_r": 30,
                "margin_v": 120,
                "effect": "{\\fad(300,300)\\t(0,500,\\fscy120)\\t(500,1000,\\fscy100)}"
            },
            "Popline": {
                "font": "Impact",
                "font_size": 56,
                "primary_color": "&H00FF9900",
                "outline_color": "&H00000000",
                "back_color": "&H80000000",
                "bold": -1,
                "italic": 0,
                "border": 2,
                "outline": 3,
                "shadow": 2,
                "alignment": 8,
                "margin_l": 20,
                "margin_r": 20,
                "margin_v": 80,
                "effect": "{\\fad(200,200)\\t(0,500,\\fscx120\\fscy120)\\t(500,1000,\\fscx100\\fscy100)}"
            },
            "Simple": {
                "font": "Segoe UI",
                "font_size": 42,
                "primary_color": "&H00FFFFFF",
                "outline_color": "&H00000000",
                "back_color": "&H80000000",
                "bold": 0,
                "italic": 0,
                "border": 1,
                "outline": 1,
                "shadow": 0,
                "alignment": 2,
                "margin_l": 30,
                "margin_r": 30,
                "margin_v": 100,
                "effect": "{\\fad(200,200)}"
            },
            "Hormozi": {
                "font": "Arial Black",
                "font_size": 60,
                "primary_color": "&H0000FFFF",
                "outline_color": "&H00000000",
                "back_color": "&H80000000",
                "bold": -1,
                "italic": 0,
                "border": 3,
                "outline": 4,
                "shadow": 2,
                "alignment": 2,
                "margin_l": 40,
                "margin_r": 40,
                "margin_v": 150,
                "effect": "{\\fad(300,300)\\t(0,1000,\\c&HFFFFFF&\\t(1000,2000,\\c&H00FFFF&)}"
            },
            "MrBeast": {
                "font": "Bahnschrift",
                "font_size": 64,
                "primary_color": "&H0000FF00",
                "outline_color": "&H00000000",
                "back_color": "&H80000000",
                "bold": -1,
                "italic": 0,
                "border": 4,
                "outline": 5,
                "shadow": 3,
                "alignment": 2,
                "margin_l": 50,
                "margin_r": 50,
                "margin_v": 180,
                "effect": "{\\fad(400,400)\\t(0,1000,\\fscx150\\fscy150)\\t(1000,2000,\\fscx100\\fscy100)}"
            },
            "Dreamy": {
                "font": "Segoe UI Light",
                "font_size": 48,
                "primary_color": "&H00FFD700",  # Gold
                "outline_color": "&H00663399",  # Purple
                "back_color": "&H40000000",
                "bold": 0,
                "italic": 1,
                "border": 2,
                "outline": 3,
                "shadow": 1,
                "alignment": 2,
                "margin_l": 40,
                "margin_r": 40,
                "margin_v": 150,
                "effect": "{\\fad(500,500)\\t(0,1500,\\c&HFFD700&\\blur3)\\t(1500,3000,\\c&HFFFFFF&\\blur0)}"
            },
            "Tech": {
                "font": "Consolas",
                "font_size": 52,
                "primary_color": "&H0000FFFF",  # Cyan
                "outline_color": "&H00003366",  # Dark blue
                "back_color": "&H00000000",
                "bold": -1,
                "italic": 0,
                "border": 3,
                "outline": 4,
                "shadow": 2,
                "alignment": 1,  # Left
                "margin_l": 60,
                "margin_r": 30,
                "margin_v": 100,
                "effect": "{\\fad(300,300)\\t(0,1000,\\fscx110\\fscy110\\1c&H00FF00&)\\t(1000,2000,\\fscx100\\fscy100\\1c&H00FFFF&)}"
            },
            "Elegant": {
                "font": "Garamond",
                "font_size": 56,
                "primary_color": "&H00FFFFFF",  # White
                "outline_color": "&H00663333",  # Dark brown
                "back_color": "&H20000000",
                "bold": 0,
                "italic": 1,
                "border": 1,
                "outline": 2,
                "shadow": 1,
                "alignment": 8,  # Bottom center
                "margin_l": 30,
                "margin_r": 30,
                "margin_v": 50,
                "effect": "{\\fad(400,400)\\t(0,1000,\\alpha&H88&)\\t(1000,2000,\\alpha&H00&)}"
            },
            "Gaming": {
                "font": "Bauhaus 93",
                "font_size": 60,
                "primary_color": "&H00FF6600",  # Orange
                "outline_color": "&H00000000",  # Black
                "back_color": "&H60000000",
                "bold": -1,
                "italic": 0,
                "border": 5,
                "outline": 6,
                "shadow": 4,
                "alignment": 2,
                "margin_l": 40,
                "margin_r": 40,
                "margin_v": 120,
                "effect": "{\\fad(200,200)\\t(0,500,\\fscx120\\fscy120\\c&HFF0000&)\\t(500,1000,\\fscx100\\fscy100\\c&HFF6600&)}"
            },
            "News": {
                "font": "Arial",
                "font_size": 50,
                "primary_color": "&H00FFFFFF",  # White
                "outline_color": "&H00000000",  # Black
                "back_color": "&H80000000",
                "bold": -1,
                "italic": 0,
                "border": 2,
                "outline": 3,
                "shadow": 2,
                "alignment": 2,
                "margin_l": 30,
                "margin_r": 30,
                "margin_v": 160,
                "effect": "{\\fad(300,300)\\t(0,1000,\\move(1080,1920,540,1500))}"
            },
            "Comic": {
                "font": "Comic Sans MS",
                "font_size": 54,
                "primary_color": "&H000000FF",  # Blue
                "outline_color": "&H00FFFFFF",  # White
                "back_color": "&H20FFC0CB",    # Pink tint
                "bold": 0,
                "italic": 0,
                "border": 4,
                "outline": 5,
                "shadow": 3,
                "alignment": 2,
                "margin_l": 40,
                "margin_r": 40,
                "margin_v": 140,
                "effect": "{\\fad(300,300)\\t(0,1000,\\frz5\\frx5)\\t(1000,2000,\\frz0\\frx0)}"
            },
            "Horror": {
                "font": "Chiller",
                "font_size": 58,
                "primary_color": "&H00FF0000",  # Red
                "outline_color": "&H00000000",  # Black
                "back_color": "&H40000000",
                "bold": -1,
                "italic": 0,
                "border": 3,
                "outline": 4,
                "shadow": 3,
                "alignment": 2,
                "margin_l": 50,
                "margin_r": 50,
                "margin_v": 170,
                "effect": "{\\fad(300,300)\\t(0,1000,\\blur3\\c&H880000&)\\t(1000,2000,\\blur0.5\\c&HFF0000&)}"
            },
            "Corporate": {
                "font": "Calibri",
                "font_size": 48,
                "primary_color": "&H00003399",  # Dark blue
                "outline_color": "&H00FFFFFF",  # White
                "back_color": "&H20FFFFFF",
                "bold": -1,
                "italic": 0,
                "border": 1,
                "outline": 2,
                "shadow": 1,
                "alignment": 1,  # Left
                "margin_l": 80,
                "margin_r": 30,
                "margin_v": 100,
                "effect": "{\\fad(400,400)\\t(0,1000,\\alpha&HAA&)\\t(1000,2000,\\alpha&H00&)}"
            },
            "SocialMedia": {
                "font": "Helvetica",
                "font_size": 52,
                "primary_color": "&H00FFFFFF",  # White
                "outline_color": "&H00FF00FF",  # Pink
                "back_color": "&H00000000",
                "bold": 0,
                "italic": 0,
                "border": 2,
                "outline": 3,
                "shadow": 2,
                "alignment": 2,
                "margin_l": 30,
                "margin_r": 30,
                "margin_v": 130,
                "effect": "{\\fad(300,300)\\t(0,1000,\\fscx110\\fscy110\\c&HFF00FF&)\\t(1000,2000,\\fscx100\\fscy100\\c&HFFFFFF&)}"
            },
            "Retro": {
                "font": "Courier New",
                "font_size": 50,
                "primary_color": "&H0000FF00",  # Green
                "outline_color": "&H00000000",  # Black
                "back_color": "&H00000000",
                "bold": -1,
                "italic": 0,
                "border": 1,
                "outline": 2,
                "shadow": 1,
                "alignment": 2,
                "margin_l": 40,
                "margin_r": 40,
                "margin_v": 120,
                "effect": "{\\fad(300,300)\\t(0,1000,\\alpha&HAA&\\fry10)\\t(1000,2000,\\alpha&H00&\\fry0)}"
            },
            "Futuristic": {
                "font": "Eurostile",
                "font_size": 56,
                "primary_color": "&H0000FFFF",  # Cyan
                "outline_color": "&H00000088",  # Dark blue
                "back_color": "&H20000000",
                "bold": -1,
                "italic": 0,
                "border": 3,
                "outline": 4,
                "shadow": 3,
                "alignment": 2,
                "margin_l": 40,
                "margin_r": 40,
                "margin_v": 150,
                "effect": "{\\fad(300,300)\\t(0,1000,\\blur2\\fscx115\\fscy115)\\t(1000,2000,\\blur0\\fscx100\\fscy100)}"
            }
        }
    
    def process_json_clips(self, json_path):
        """Process video clips based on JSON configuration"""
        try:
            # Load JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            self.status_updated.emit(f"Processing {config['video_title']}")
            self.log_updated.emit(f"📋 Processing JSON config for: {config['video_title']}")
            
            # Create output directory
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Open video file
            video = VideoFileClip(self.video_path)
            total_clips = len(config['clips'])
            processed_clips = 0
            
            for clip_info in config['clips']:
                # Parse timestamp (format: "02:53 - 03:30")
                start_str, end_str = clip_info['timestamp'].split('-')
                start = self.parse_time_to_seconds(start_str.strip())
                end = self.parse_time_to_seconds(end_str.strip())
                
                self.status_updated.emit(f"Creating clip: {clip_info['clip_title']}")
                self.log_updated.emit(f"✂️ Cutting clip: {clip_info['clip_title']} ({start}s-{end}s)")
                
                # Create clip
                clip = video.subclip(start, end)
                
                if self.vertical_format:
                    clip = self.convert_to_vertical(clip)
                
                # Create filename from clip title (safe for filesystem)
                safe_title = "".join(c if c.isalnum() else "_" for c in clip_info['clip_title'])
                clip_filename = f"{processed_clips+1:02d}_{safe_title[:50]}.mp4"
                clip_path = os.path.join(self.output_dir, clip_filename)
                
                # Write clip
                clip.write_videofile(
                    clip_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True,
                    verbose=False,
                    logger=None
                )
                
                # Add subtitles if enabled
                if self.add_subtitles:
                    self.status_updated.emit(f"Adding subtitles to {clip_info['clip_title']}")
                    self.log_updated.emit(f"📝 Generating subtitles for {clip_filename}...")
                    subtitle_path = self.generate_subtitle(clip_path)
                    if subtitle_path:
                        final_path = clip_path.replace(".mp4", "_with_subs.mp4")
                        self.add_subtitle_to_video(clip_path, subtitle_path, final_path)
                        os.remove(clip_path)
                        os.rename(final_path, clip_path)
                
                processed_clips += 1
                progress = int(100 * processed_clips / total_clips)
                self.progress_updated.emit(progress)
                self.log_updated.emit(f"✅ Completed clip {processed_clips}/{total_clips}")
            
            video.close()
            self.log_updated.emit(f"🎉 Successfully processed {processed_clips} clips from JSON!")
            return True, f"Created {processed_clips} clips from JSON config"
            
        except Exception as e:
            self.log_updated.emit(f"❌ JSON processing error: {str(e)}")
            return False, f"Error processing JSON: {str(e)}"
    
    def parse_time_to_seconds(self, time_str):
        """Convert timestamp string (MM:SS or HH:MM:SS) to seconds"""
        parts = list(map(float, time_str.split(':')))
        if len(parts) == 3:  # HH:MM:SS
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:  # MM:SS
            return parts[0] * 60 + parts[1]
        else:
            return float(parts[0])  # SS

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🎬 C1PP3R_PY")
        self.setGeometry(100, 100, 900, 700)
        self.worker = None
        self.setup_modern_ui()
        self.apply_modern_theme()
    
    def setup_modern_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 30, 30, 30)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(20)
        
        # Input Section
        input_group = self.create_input_section()
        auto_group = QGroupBox("🤖 Auto Recommendations")
        auto_group.setFont(QFont("Segoe UI", 12, QFont.Bold))
        auto_layout = QHBoxLayout()
        
        self.auto_btn = ModernButton("✨ Auto Recommend Clips", primary=True)
        self.auto_btn.clicked.connect(self.auto_recommend)
        auto_layout.addWidget(self.auto_btn)
        
        auto_group.setLayout(auto_layout)
        content_layout.insertWidget(1, auto_group)  
        content_layout.addWidget(input_group)
        
        
        # Settings Section
        settings_group = self.create_settings_section()
        content_layout.addWidget(settings_group)
        
        # Progress Section
        progress_group = self.create_progress_section()
        content_layout.addWidget(progress_group)
        
        # Log Section
        log_group = self.create_log_section()
        content_layout.addWidget(log_group)
        
        content_widget.setLayout(content_layout)
        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)
        
        # Process Button
        self.process_btn = ModernButton("🚀 Start Processing", primary=True)
        self.process_btn.setMinimumHeight(50)
        self.process_btn.clicked.connect(self.process_video)
        main_layout.addWidget(self.process_btn)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Add JSON button to input section
        json_layout = QHBoxLayout()
        self.json_input = ModernLineEdit("Select JSON config file")
        self.json_browse_btn = ModernButton("📝 Browse Config (JSON)")
        self.json_browse_btn.clicked.connect(self.browse_json)
        self.json_process_btn = ModernButton("🔄 Process", primary=True)
        self.json_process_btn.clicked.connect(self.process_json)
        
        json_layout.addWidget(self.json_input, 4)
        json_layout.addWidget(self.json_browse_btn, 1)
        json_layout.addWidget(self.json_process_btn, 2)
        
        # Add this layout to your existing input_group layout
        input_group_layout = input_group.layout()
        input_group_layout.addLayout(json_layout)

    def auto_recommend(self):
        """Handle tombol rekomendasi otomatis"""
        source_text = self.source_input.text().strip()
        output_text = self.output_input.text().strip()
        
        if not source_text:
            QMessageBox.warning(self, "⚠️ Warning", "Please enter a YouTube URL or select a video file!")
            return
        
        if not output_text:
            QMessageBox.warning(self, "⚠️ Warning", "Please select an output directory!")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "⚠️ Warning", "A process is already running!")
            return
        
        # Setup worker
        self.progress_bar.setValue(0)
        self.status_label.setText("Finding interesting moments...")
        self.auto_btn.setEnabled(False)
        self.auto_btn.setText("🔍 Analyzing...")
        
        self.worker = Worker()
        self.worker.video_path = source_text
        self.worker.output_dir = output_text
        self.worker.add_subtitles = self.subtitle_check.isChecked()
        self.worker.whisper_model = self.model_combo.currentText().split()[0]
        self.worker.vertical_format = self.vertical_check.isChecked()
        self.worker.is_youtube_url = source_text.startswith(('http://', 'https://'))
        self.worker.subtitle_style = self.style_combo.currentText()
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.log_updated.connect(self.update_log)
        self.worker.finished.connect(self.auto_recommend_finished)
        
        # Start finding interesting clips
        self.worker.run_auto_recommend = True
        self.worker.start()

    def auto_recommend_finished(self, success, message):
        """Handle selesainya proses rekomendasi otomatis"""
        self.auto_btn.setEnabled(True)
        self.auto_btn.setText("✨ Auto Recommend Clips")
        
        if success:
            self.status_label.setText("✅ Found interesting moments!")
            QMessageBox.information(self, "🎉 Success", message)
        else:
            self.status_label.setText("❌ " + message)
            QMessageBox.critical(self, "⚠️ Error", message)
    
    def create_header(self):
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 15px;
                color: white;
                padding: 20px;
            }
        """)
        
        layout = QVBoxLayout()
        
        title = QLabel("🎬 C1PP3R_PY")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: white; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        
        subtitle = QLabel("Transform long videos into engaging short clips\n Thank you to ChatGPT, DeepSeek, Gemini, and Claude\n Made with ♡")
        subtitle.setFont(QFont("Segoe UI", 12))
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.8); background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(title)
        layout.addWidget(subtitle)
        header_frame.setLayout(layout)
        
        return header_frame
    
    def create_input_section(self):
        group = QGroupBox("📁 Input Configuration")
        group.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout = QVBoxLayout()
        
        # Video Source
        source_layout = QHBoxLayout()
        source_label = QLabel("Video Source:")
        source_label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.source_input = ModernLineEdit("Enter YouTube URL or select local video file")
        self.browse_btn = ModernButton("📂 Browse")
        self.browse_btn.clicked.connect(self.browse_file)
        
        source_layout.addWidget(source_label, 1)
        source_layout.addWidget(self.source_input, 4)
        source_layout.addWidget(self.browse_btn, 1)
        
        # Output Directory
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Folder:")
        output_label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.output_input = ModernLineEdit("Select where to save the clips")
        self.output_browse_btn = ModernButton("📁 Browse")
        self.output_browse_btn.clicked.connect(self.browse_output)
        
        output_layout.addWidget(output_label, 1)
        output_layout.addWidget(self.output_input, 4)
        output_layout.addWidget(self.output_browse_btn, 1)
        
        layout.addLayout(source_layout)
        layout.addLayout(output_layout)
        group.setLayout(layout)
        
        return group
    
    def create_settings_section(self):
        group = QGroupBox("⚙️ Processing Settings")
        group.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout = QGridLayout()
        layout.setVerticalSpacing(15)
        layout.setHorizontalSpacing(20)
        layout.setContentsMargins(15, 15, 15, 15)

        # Row 0: Clip Duration and AI Model Quality
        duration_label = QLabel("Clip Duration:")
        duration_label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.duration_spin = ModernSpinBox()
        self.duration_spin.setRange(5, 300)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" seconds")

        model_label = QLabel("AI Model Quality:")
        model_label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.model_combo = ModernComboBox()
        self.model_combo.addItems(["tiny (fastest)", "base (recommended)", "small", "medium", "large (best quality)"])
        self.model_combo.setCurrentIndex(1)

        layout.addWidget(duration_label, 0, 0)
        layout.addWidget(self.duration_spin, 0, 1)
        layout.addWidget(model_label, 0, 2)
        layout.addWidget(self.model_combo, 0, 3)

        # Row 1: Subtitle Style (full width)
        style_label = QLabel("Sub Styles")
        style_label.setFont(QFont("Segoe UI", 10, QFont.Medium))
        self.style_combo = ModernComboBox()
        self.style_combo.addItems(["Karaoke", "Popline", "Simple", "Hormozi", "MrBeast", "Dreamy", "Tech", "Elegant", "Gaming", "News", "Comic", "Horror", "Corporate", "SocialMedia", "Retro", "Futuristic"])
        
        layout.addWidget(style_label, 1, 0)
        layout.addWidget(self.style_combo, 1, 1, 1, 3)  # Span across 3 columns

        # Row 2: Checkboxes (properly organized)
        self.subtitle_check = ModernCheckBox("Add Subtitle")
        self.subtitle_check.setChecked(True)        
        self.vertical_check = ModernCheckBox("📱 Convert to Vertical Format (9:16)")
        self.vertical_check.setChecked(True)

        # Organize checkboxes in a horizontal layout
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.subtitle_check)
        checkbox_layout.addStretch()  # Push vertical format to the right
        checkbox_layout.addWidget(self.vertical_check)
        
        layout.addLayout(checkbox_layout, 2, 0, 1, 4)  # Span all columns

        group.setLayout(layout)
        return group
    
    def create_progress_section(self):
        group = QGroupBox("📊 Progress")
        group.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout = QVBoxLayout()
        
        self.progress_bar = ModernProgressBar()
        self.progress_bar.setRange(0, 100)
        
        self.status_label = QLabel("Ready to process")
        self.status_label.setFont(QFont("Segoe UI", 11))
        self.status_label.setStyleSheet("color: #6c757d; padding: 5px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        group.setLayout(layout)
        
        return group
    
    def create_log_section(self):
        group = QGroupBox("📝 Processing Log")
        group.setFont(QFont("Segoe UI", 12, QFont.Bold))
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #f8f9fa;
                border: 2px solid #e9ecef;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        self.log_text.setPlaceholderText("Processing logs will appear here...")
        
        layout.addWidget(self.log_text)
        group.setLayout(layout)
        
        return group
    
    def apply_modern_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background: #ffffff;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #e9ecef;
                border-radius: 10px;
                margin-top: 1ex;
                padding-top: 15px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #495057;
                background: white;
            }
            QLabel {
                color: #495057;
            }
        """)
    
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*)"
        )
        if file_path:
            self.source_input.setText(file_path)
    
    def browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_input.setText(dir_path)

    def process_video(self):
        source_text = self.source_input.text().strip()
        output_text = self.output_input.text().strip()
        
        if not source_text:
            QMessageBox.warning(self, "⚠️ Warning", "Please enter a YouTube URL or select a video file!")
            return
        
        if not output_text:
            QMessageBox.warning(self, "⚠️ Warning", "Please select an output directory!")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "⚠️ Warning", "A process is already running!")
            return
        
        # Clear log
        self.log_text.clear()
        
        # Setup worker
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing...")
        self.process_btn.setEnabled(False)
        self.process_btn.setText("🔄 Processing...")
        
        self.worker = Worker()
        self.worker.video_path = source_text
        self.worker.output_dir = output_text
        self.worker.clip_duration = self.duration_spin.value()
        self.worker.whisper_model = self.model_combo.currentText().split()[0]  # Extract model name
        self.worker.add_subtitles = self.subtitle_check.isChecked()
        self.worker.vertical_format = self.vertical_check.isChecked()
        self.worker.is_youtube_url = source_text.startswith(('http://', 'https://'))
        self.worker.subtitle_style = self.style_combo.currentText()
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.log_updated.connect(self.update_log)
        self.worker.finished.connect(self.process_finished)
        
        # Start the worker
        self.worker.start()

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        
    def update_status(self, text):
        self.status_label.setText(text)
        
    def update_log(self, text):
        self.log_text.append(text)
        
    def process_finished(self, success, message):
        self.process_btn.setEnabled(True)
        self.process_btn.setText("🚀 Start Processing")
        
        if success:
            self.status_label.setText("✅ " + message)
            QMessageBox.information(self, "🎉 Success", message)
        else:
            self.status_label.setText("❌ " + message)
            QMessageBox.critical(self, "⚠️ Error", message)
            
        # Enable the process button
        self.process_btn.setEnabled(True)
        self.process_btn.setText("🚀 Start Processing")

    def closeEvent(self, event):
        """Handle window close event"""
        if self.worker and self.worker.isRunning():
            reply = QMessageBox.question(
                self, '⚠️ Process Running',
                'A video processing task is still running. Are you sure you want to quit?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.worker.terminate()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def browse_json(self):
        """Browse for JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON File", "", 
            "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.json_input.setText(file_path)
    
    def process_json(self):
        """Process video based on JSON config"""
        json_path = self.json_input.text().strip()
        source_text = self.source_input.text().strip()
        output_text = self.output_input.text().strip()
        
        if not json_path:
            QMessageBox.warning(self, "⚠️ Warning", "Please select a JSON config file!")
            return
        
        if not source_text:
            QMessageBox.warning(self, "⚠️ Warning", "Please enter a YouTube URL or select a video file!")
            return
        
        if not output_text:
            QMessageBox.warning(self, "⚠️ Warning", "Please select an output directory!")
            return
        
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "⚠️ Warning", "A process is already running!")
            return
        
        # Clear log
        self.log_text.clear()
        
        # Setup worker
        self.progress_bar.setValue(0)
        self.status_label.setText("Initializing JSON processing...")
        self.json_process_btn.setEnabled(False)
        self.json_process_btn.setText("🔄 Processing JSON...")
        
        self.worker = Worker()
        self.worker.video_path = source_text
        self.worker.output_dir = output_text
        self.worker.whisper_model = self.model_combo.currentText().split()[0]
        self.worker.add_subtitles = self.subtitle_check.isChecked()
        self.worker.vertical_format = self.vertical_check.isChecked()
        self.worker.is_youtube_url = source_text.startswith(('http://', 'https://'))
        self.worker.subtitle_style = self.style_combo.currentText()
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.update_status)
        self.worker.log_updated.connect(self.update_log)
        
        # Create and start JSON processing thread
        self.json_thread = JsonProcessingThread(self.worker, json_path)
        self.json_thread.finished_signal.connect(self.json_process_finished)
        self.json_thread.start()
    
    def json_process_finished(self, success, message):
        """Handle completion of JSON processing"""
        self.json_process_btn.setEnabled(True)
        self.json_process_btn.setText("🔄 Process JSON")
        
        if success:
            self.status_label.setText("✅ " + message)
            QMessageBox.information(self, "🎉 Success", message)
        else:
            self.status_label.setText("❌ " + message)
            QMessageBox.critical(self, "⚠️ Error", message)
        
        # Clean up
        self.json_thread.quit()
        self.json_thread.wait()
        self.json_thread.deleteLater()

def main():
    app = QApplication(sys.argv)
    
    # Set application style and palette
    app.setStyle('Fusion')
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()