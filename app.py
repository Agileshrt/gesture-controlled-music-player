import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import threading
import pygame
import pyautogui
import os
from collections import deque
from PIL import Image, ImageTk, ImageDraw
import time
import math

# ----------------------------- Settings -----------------------------
MODEL_PATH = "models/gesture_model_conv1d_best.h5"
LABELS = list(np.load("label_map.npy"))  # ['play','pause','next','previous','volume_up','volume_down']
SEQ_LENGTH = 30
CONF_THRESHOLD = 0.85

model = load_model(MODEL_PATH)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

pygame.mixer.init()
pygame.mixer.music.set_volume(0.5)

SONG_FOLDER = "songs"
song_list = [f for f in os.listdir(SONG_FOLDER) if f.endswith(".mp3")]
song_index = 0

gesture_history = deque(maxlen=5)
volume_level = 50
current_gesture = "None"
last_action_time = 0
ACTION_COOLDOWN = 1.5  # seconds

# Animation variables
pulse_radius = 0
pulse_growing = True
visualization_angle = 0
hologram_effect = 0
particles = []

# ----------------------------- Preprocess function -----------------------------
def preprocess_sequence(seq_list):
    arr = np.array(seq_list, dtype=np.float32).reshape(SEQ_LENGTH, 21, 3)
    root = arr[:, 0:1, :]
    arr_rel = arr - root
    max_xy = max(np.max(np.abs(arr_rel[:, :, 0:2])), 1e-6)
    arr_rel[:, :, 0:2] /= max_xy
    arr_rel[:, :, 2] /= (max_xy + 1e-6)
    arr_flat = arr_rel.reshape(SEQ_LENGTH, 63)
    arr_flat = (arr_flat - arr_flat.mean()) / (arr_flat.std() + 1e-6)
    return arr_flat.reshape(1, SEQ_LENGTH, 63)

# ----------------------------- Modern Webcam Animation Functions -----------------------------
def create_modern_frame(image, frame_count):
    """Add modern sci-fi style frame and effects to webcam image"""
    height, width = image.shape[:2]
    
    # Create modern gradient border
    border_size = 40
    
    # Create gradient border
    bordered = cv2.copyMakeBorder(
        image, 
        border_size, border_size, border_size, border_size, 
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    # Add inner glow effect
    glow_size = 5
    bordered = cv2.copyMakeBorder(
        bordered,
        glow_size, glow_size, glow_size, glow_size,
        cv2.BORDER_CONSTANT, value=(30, 144, 255)  # Dodger blue
    )
    
    # Add outer frame with gradient
    outer_border = 15
    final_frame = cv2.copyMakeBorder(
        bordered,
        outer_border, outer_border, outer_border, outer_border,
        cv2.BORDER_CONSTANT, value=(25, 25, 112)  # Midnight blue
    )
    
    # Add corner accents
    corner_length = 60
    thickness = 4
    color = (30, 144, 255)  # Dodger blue
    
    # Top-left corner
    cv2.line(final_frame, (20, 20), (20 + corner_length, 20), color, thickness)
    cv2.line(final_frame, (20, 20), (20, 20 + corner_length), color, thickness)
    
    # Top-right corner
    cv2.line(final_frame, (final_frame.shape[1] - 20, 20), 
             (final_frame.shape[1] - 20 - corner_length, 20), color, thickness)
    cv2.line(final_frame, (final_frame.shape[1] - 20, 20), 
             (final_frame.shape[1] - 20, 20 + corner_length), color, thickness)
    
    # Bottom-left corner
    cv2.line(final_frame, (20, final_frame.shape[0] - 20), 
             (20 + corner_length, final_frame.shape[0] - 20), color, thickness)
    cv2.line(final_frame, (20, final_frame.shape[0] - 20), 
             (20, final_frame.shape[0] - 20 - corner_length), color, thickness)
    
    # Bottom-right corner
    cv2.line(final_frame, (final_frame.shape[1] - 20, final_frame.shape[0] - 20), 
             (final_frame.shape[1] - 20 - corner_length, final_frame.shape[0] - 20), color, thickness)
    cv2.line(final_frame, (final_frame.shape[1] - 20, final_frame.shape[0] - 20), 
             (final_frame.shape[1] - 20, final_frame.shape[0] - 20 - corner_length), color, thickness)
    
    # Add hologram scan effect
    global hologram_effect
    scan_line_y = (hologram_effect + border_size + glow_size + outer_border) % final_frame.shape[0]
    scan_height = 3
    
    # Create scan line with gradient
    for i in range(scan_height):
        alpha = 1.0 - (i / scan_height)
        line_y = scan_line_y + i
        if line_y < final_frame.shape[0]:
            overlay = final_frame.copy()
            cv2.line(overlay, (0, line_y), (final_frame.shape[1], line_y), 
                    (0, 255, 255, 100), 1)  # Cyan color
            final_frame = cv2.addWeighted(overlay, alpha * 0.3, final_frame, 1 - alpha * 0.3, 0)
    
    hologram_effect = (hologram_effect + 2) % final_frame.shape[0]
    
    # Add particle effects when gesture is detected
    if current_gesture != "None" and current_gesture != "Collecting...":
        particle_color = (0, 255, 255)  # Cyan
        num_particles = 5
        
        for _ in range(num_particles):
            x = np.random.randint(border_size, final_frame.shape[1] - border_size)
            y = np.random.randint(border_size, final_frame.shape[0] - border_size)
            radius = np.random.randint(1, 3)
            cv2.circle(final_frame, (x, y), radius, particle_color, -1)
    
    # Add status indicator
    if current_gesture != "None" and current_gesture != "Collecting...":
        status_color = (0, 255, 0)  # Green
        status_text = "GESTURE DETECTED"
    else:
        status_color = (255, 255, 0)  # Yellow
        status_text = "AWAITING GESTURE"
    
    # Add status bar at bottom
    status_bar_height = 30
    status_bar = np.zeros((status_bar_height, final_frame.shape[1], 3), dtype=np.uint8)
    status_bar[:] = (40, 40, 40)  # Dark gray
    
    # Add animated dots to status bar
    dot_spacing = 15
    num_dots = final_frame.shape[1] // dot_spacing
    for i in range(num_dots):
        dot_x = (i * dot_spacing + frame_count) % final_frame.shape[1]
        dot_radius = 2 + int(1.5 * math.sin(frame_count * 0.1 + i * 0.5))
        cv2.circle(status_bar, (dot_x, status_bar_height // 2), dot_radius, status_color, -1)
    
    # Add status text
    text_x = final_frame.shape[1] // 2 - len(status_text) * 6
    cv2.putText(status_bar, status_text, (text_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 
               0.5, status_color, 1)
    
    # Combine status bar with main frame
    final_frame = np.vstack([final_frame, status_bar])
    
    return final_frame

def update_visualization():
    """Update background visualization animations"""
    global pulse_radius, pulse_growing, visualization_angle
    
    # Pulse animation for gesture display
    if pulse_growing:
        pulse_radius += 1
        if pulse_radius > 10:
            pulse_growing = False
    else:
        pulse_radius -= 1
        if pulse_radius < 0:
            pulse_growing = True
    
    # Rotating visualization angle
    visualization_angle = (visualization_angle + 2) % 360
    
    # Update visualization canvas
    visualization_canvas.delete("all")
    
    # Create animated background
    width = visualization_canvas.winfo_width()
    height = visualization_canvas.winfo_height()
    
    if width > 1 and height > 1:  # Only draw if canvas is initialized
        center_x, center_y = width // 2, height // 2
        max_radius = min(center_x, center_y) - 20
        
        # Draw rotating arcs with gradient colors
        colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12']
        for i, color in enumerate(colors):
            angle = (visualization_angle + i * 72) % 360
            rad_angle = math.radians(angle)
            x1 = center_x + (max_radius - 30) * math.cos(rad_angle)
            y1 = center_y + (max_radius - 30) * math.sin(rad_angle)
            x2 = center_x + max_radius * math.cos(rad_angle)
            y2 = center_y + max_radius * math.sin(rad_angle)
            
            visualization_canvas.create_line(x1, y1, x2, y2, 
                                           fill=color, width=3, 
                                           capstyle=tk.ROUND)
    
    root.after(50, update_visualization)

# ----------------------------- User-Friendly Gesture Guide -----------------------------
def show_gesture_guide():
    guide_win = tk.Toplevel(root)
    guide_win.title("🎭 Gesture Control Guide")
    guide_win.geometry("900x700")
    guide_win.configure(bg='#1e2a36')
    guide_win.resizable(False, False)
    
    # Center the window
    guide_win.transient(root)
    guide_win.grab_set()
    
    # Header
    header_frame = tk.Frame(guide_win, bg='#1e2a36')
    header_frame.pack(fill='x', pady=(25, 15))
    
    tk.Label(header_frame, text="🎭 Gesture Control Guide", 
            font=("Arial", 26, "bold"), fg="white", bg="#1e2a36").pack()
    
    tk.Label(header_frame, text="Learn how to control music with simple hand gestures", 
            font=("Arial", 14), fg="#bdc3c7", bg="#1e2a36").pack(pady=8)
    
    # Create notebook for tabbed interface
    notebook = ttk.Notebook(guide_win)
    notebook.pack(fill='both', expand=True, padx=25, pady=15)
    
    # Style the notebook
    style = ttk.Style()
    style.configure("TNotebook", background='#1e2a36')
    style.configure("TNotebook.Tab", background='#34495e', foreground='white', font=('Arial', 11, 'bold'))
    style.map("TNotebook.Tab", background=[('selected', '#3498db')])
    
    # Basic Controls Tab
    basic_frame = tk.Frame(notebook, bg='#2c3e50')
    notebook.add(basic_frame, text="🎵 Basic Controls")
    
    # Playback Controls Tab
    playback_frame = tk.Frame(notebook, bg='#2c3e50')
    notebook.add(playback_frame, text="⏯️ Playback")
    
    # Volume Controls Tab
    volume_frame = tk.Frame(notebook, bg='#2c3e50')
    notebook.add(volume_frame, text="🔊 Volume")
    
    # Fill the tabs with content
    fill_basic_controls(basic_frame)
    fill_playback_controls(playback_frame)
    fill_volume_controls(volume_frame)
    
    # Quick Tips Section
    tips_frame = tk.Frame(guide_win, bg='#34495e', relief='raised', bd=2)
    tips_frame.pack(fill='x', padx=25, pady=(10, 20), ipady=10)
    
    tk.Label(tips_frame, text="💡 Quick Tips for Best Results", 
            font=("Arial", 14, "bold"), fg="#f39c12", bg='#34495e').pack(pady=8)
    
    tips_grid = tk.Frame(tips_frame, bg='#34495e')
    tips_grid.pack(padx=20, pady=5)
    
    tips = [
        ("✨ Good Lighting", "Ensure even lighting on your hand"),
        ("📏 Steady Hand", "Hold gesture for 2-3 seconds"),
        ("🎯 Clear Background", "Avoid cluttered backgrounds"),
        ("📷 Full View", "Keep hand within camera frame"),
        ("⏱️ Be Patient", "Wait for confirmation feedback"),
        ("🔧 Practice", "Try each gesture few times")
    ]
    
    for i, (title, desc) in enumerate(tips):
        row = i // 3
        col = i % 3
        
        tip_frame = tk.Frame(tips_grid, bg='#2c3e50', relief='ridge', bd=1)
        tip_frame.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
        
        tk.Label(tip_frame, text=title, font=("Arial", 11, "bold"), 
                fg="#3498db", bg='#2c3e50').pack(pady=(8, 2))
        tk.Label(tip_frame, text=desc, font=("Arial", 9), 
                fg="#ecf0f1", bg='#2c3e50', wraplength=200).pack(pady=(2, 8))
    
    # Close button
    close_btn = tk.Button(guide_win, text="🎯 Start Controlling Now!", 
                         font=("Arial", 14, "bold"), command=guide_win.destroy, 
                         bg='#27ae60', fg='white', relief='raised', 
                         padx=30, pady=12, bd=3)
    close_btn.pack(pady=20)

def fill_basic_controls(parent):
    """Fill basic controls tab"""
    content_frame = tk.Frame(parent, bg='#2c3e50')
    content_frame.pack(fill='both', expand=True, padx=20, pady=20)
    
    instructions = [
        "Welcome to Hand Gesture Music Control!",
        "This system allows you to control music playback using simple hand gestures.",
        "No buttons to click, no keyboard needed - just show your hand to the camera!",
        "",
        "How it works:",
        "• Show your hand clearly to the camera",
        "• Hold the gesture for 2-3 seconds", 
        "• Wait for visual confirmation",
        "• The system will perform the action automatically",
        "",
        "Switch between tabs above to learn specific gestures."
    ]
    
    for i, text in enumerate(instructions):
        if i == 0:
            font = ("Arial", 16, "bold")
            fg = "#3498db"
        elif i == 3:
            font = ("Arial", 12, "bold")
            fg = "#f39c12"
        else:
            font = ("Arial", 11)
            fg = "#ecf0f1"
        
        label = tk.Label(content_frame, text=text, font=font, fg=fg, 
                        bg='#2c3e50', justify='left', wraplength=800)
        label.pack(anchor='w', pady=3)

def fill_playback_controls(parent):
    """Fill playback controls tab"""
    content_frame = tk.Frame(parent, bg='#2c3e50')
    content_frame.pack(fill='both', expand=True, padx=20, pady=20)
    
    controls = [
        ("✋ Open Palm", "PLAY", "Starts music playback", "#2ecc71"),
        ("✊ Closed Fist", "PAUSE/RESUME", "Pauses or resumes current track", "#e74c3c"),
        ("☝️ Index Finger", "PREVIOUS", "Goes to previous track", "#9b59b6"), 
        ("✌️ Two Fingers", "NEXT", "Skips to next track", "#9b59b6")
    ]
    
    for gesture, action, description, color in controls:
        item_frame = tk.Frame(content_frame, bg='#34495e', relief='raised', bd=2)
        item_frame.pack(fill='x', pady=10, ipady=12)
        
        # Main content
        main_frame = tk.Frame(item_frame, bg='#34495e')
        main_frame.pack(fill='x', padx=20, pady=12)
        
        # Gesture and action
        left_frame = tk.Frame(main_frame, bg='#34495e')
        left_frame.pack(side='left', fill='y')
        
        tk.Label(left_frame, text=gesture, font=("Arial", 18, "bold"), 
                fg=color, bg='#34495e').pack(anchor='w')
        tk.Label(left_frame, text=action, font=("Arial", 14), 
                fg='#ecf0f1', bg='#34495e').pack(anchor='w', pady=(2, 0))
        
        # Description
        desc_frame = tk.Frame(main_frame, bg='#34495e')
        desc_frame.pack(side='right', fill='x', expand=True, padx=(30, 0))
        
        tk.Label(desc_frame, text=description, font=("Arial", 12), 
                fg='#bdc3c7', bg='#34495e', justify='left', wraplength=400).pack(anchor='e')

def fill_volume_controls(parent):
    """Fill volume controls tab"""
    content_frame = tk.Frame(parent, bg='#2c3e50')
    content_frame.pack(fill='both', expand=True, padx=20, pady=20)
    
    controls = [
        ("👍 Thumbs Up", "VOLUME UP", "Increases volume by 3 steps", "#27ae60"),
        ("👎 Thumbs Down", "VOLUME DOWN", "Decreases volume by 3 steps", "#e67e22")
    ]
    
    for gesture, action, description, color in controls:
        item_frame = tk.Frame(content_frame, bg='#34495e', relief='raised', bd=2)
        item_frame.pack(fill='x', pady=15, ipady=15)
        
        # Main content
        main_frame = tk.Frame(item_frame, bg='#34495e')
        main_frame.pack(fill='x', padx=25, pady=15)
        
        # Gesture icon
        tk.Label(main_frame, text=gesture, font=("Arial", 24), 
                fg=color, bg='#34495e').pack(side='left', padx=(0, 20))
        
        # Text content
        text_frame = tk.Frame(main_frame, bg='#34495e')
        text_frame.pack(side='left', fill='x', expand=True)
        
        tk.Label(text_frame, text=action, font=("Arial", 16, "bold"), 
                fg=color, bg='#34495e', anchor='w').pack(anchor='w')
        tk.Label(text_frame, text=description, font=("Arial", 12), 
                fg='#ecf0f1', bg='#34495e', anchor='w').pack(anchor='w', pady=(5, 0))

# ----------------------------- Action mapping -----------------------------
def perform_action(gesture):
    global song_index, volume_level, last_action_time
    
    current_time = time.time()
    if current_time - last_action_time < ACTION_COOLDOWN:
        return
    
    try:
        last_action_time = current_time
        
        # Add visual feedback
        show_action_feedback(gesture)
        
        if gesture == "play":
            pygame.mixer.music.load(os.path.join(SONG_FOLDER, song_list[song_index]))
            pygame.mixer.music.play()
            update_status(f"▶️ Playing: {song_list[song_index]}")
        elif gesture == "pause":
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                update_status("⏸️ Music Paused")
            else:
                pygame.mixer.music.unpause()
                update_status("▶️ Music Resumed")
        elif gesture == "next":
            song_index = (song_index + 1) % len(song_list)
            pygame.mixer.music.load(os.path.join(SONG_FOLDER, song_list[song_index]))
            pygame.mixer.music.play()
            update_status(f"⏭️ Next: {song_list[song_index]}")
        elif gesture == "previous":
            song_index = (song_index - 1) % len(song_list)
            pygame.mixer.music.load(os.path.join(SONG_FOLDER, song_list[song_index]))
            pygame.mixer.music.play()
            update_status(f"⏮️ Previous: {song_list[song_index]}")
        elif gesture == "volume_up":
            for _ in range(3):  # Multiple presses for bigger change
                pyautogui.press("volumeup")
            volume_level = min(volume_level + 15, 100)
            update_status(f"🔊 Volume Up: {volume_level}%")
            update_volume_display()
        elif gesture == "volume_down":
            for _ in range(3):  # Multiple presses for bigger change
                pyautogui.press("volumedown")
            volume_level = max(volume_level - 15, 0)
            update_status(f"🔉 Volume Down: {volume_level}%")
            update_volume_display()
    except Exception as e:
        print("Action error:", e)
        update_status("❌ Error performing action")

def show_action_feedback(gesture):
    """Show visual feedback when action is performed"""
    feedback_text = {
        "play": "▶️ PLAYING MUSIC",
        "pause": "⏸️ MUSIC PAUSED", 
        "next": "⏭️ NEXT TRACK",
        "previous": "⏮️ PREVIOUS TRACK",
        "volume_up": "🔊 VOLUME INCREASED",
        "volume_down": "🔉 VOLUME DECREASED"
    }
    
    feedback_label.config(text=feedback_text.get(gesture, ""), fg="#28a745")
    
    # Add pulse animation to feedback
    def pulse_feedback():
        for i in range(3):
            feedback_label.config(font=("Arial", 18, "bold"))
            root.update()
            time.sleep(0.1)
            feedback_label.config(font=("Arial", 16, "bold"))
            root.update()
            time.sleep(0.1)
        feedback_label.config(text="", fg="#28a745")
    
    threading.Thread(target=pulse_feedback, daemon=True).start()

def update_status(message):
    """Update status message with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    status_var.set(f"[{timestamp}] {message}")

def update_volume_display():
    """Update volume display with progress bar"""
    volume_bar['value'] = volume_level
    volume_percent.config(text=f"{volume_level}%")

# ----------------------------- GUI setup -----------------------------
root = tk.Tk()
root.title("Hand Gesture Music Player")
root.configure(bg="#1e2a36")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.geometry(f"{screen_width}x{screen_height}+0+0")
root.resizable(False, False)

# Style configuration
style = ttk.Style()
style.theme_use('clam')
style.configure("TProgressbar", thickness=20, troughcolor='#34495e', background='#3498db')

# Main container
main_frame = tk.Frame(root, bg="#1e2a36")
main_frame.pack(fill="both", expand=True, padx=20, pady=20)

# Header
header_frame = tk.Frame(main_frame, bg="#1e2a36")
header_frame.pack(fill="x", pady=(0, 20))

title_label = tk.Label(header_frame, text="🎵 Hand Gesture Music Player", 
                      font=("Arial", 28, "bold"), fg="white", bg="#1e2a36")
title_label.pack(side="left")

# Gesture guide button
gesture_btn = tk.Button(header_frame, text="🎭 Gesture Guide", font=("Arial", 14, "bold"),
                       command=show_gesture_guide, bg="#3498db", fg="white",
                       relief="flat", padx=20, pady=8)
gesture_btn.pack(side="right")

# Content area
content_frame = tk.Frame(main_frame, bg="#1e2a36")
content_frame.pack(fill="both", expand=True)

# Left panel - Info and controls
left_panel = tk.Frame(content_frame, bg="#2c3e50", width=400, relief="raised", bd=2)
left_panel.pack(side="left", fill="y", padx=(0, 20))
left_panel.pack_propagate(False)

# Visualization canvas
visualization_canvas = tk.Canvas(left_panel, bg="#1e2a36", highlightthickness=0)
visualization_canvas.place(x=0, y=0, relwidth=1, relheight=0.3)

# Current gesture display
gesture_display_frame = tk.Frame(left_panel, bg="#2c3e50")
gesture_display_frame.pack(pady=30)

gesture_display = tk.Label(gesture_display_frame, text="⬜", font=("Arial", 80), 
                          bg="#2c3e50", fg="#ecf0f1")
gesture_display.pack()

# Gesture mapping for display
gesture_display_map = {
    "play": ("▶️", "PLAY", "#2ecc71"),
    "pause": ("⏸️", "PAUSE", "#e74c3c"),
    "next": ("⏭️", "NEXT", "#3498db"),
    "previous": ("⏮️", "PREVIOUS", "#9b59b6"),
    "volume_up": ("🔊", "VOLUME UP", "#27ae60"),
    "volume_down": ("🔉", "VOLUME DOWN", "#e67e22")
}

gesture_label = tk.Label(gesture_display_frame, text="Show your hand", font=("Arial", 18, "bold"), 
                        bg="#2c3e50", fg="#bdc3c7")
gesture_label.pack(pady=5)

confidence_var = tk.DoubleVar()
confidence_bar = ttk.Progressbar(gesture_display_frame, orient="horizontal", length=300, 
                               mode="determinate", variable=confidence_var)
confidence_bar.pack(pady=10)

confidence_text = tk.Label(gesture_display_frame, text="Accuracy: 0%", font=("Arial", 12), 
                          bg="#2c3e50", fg="#bdc3c7")
confidence_text.pack()

# Song info
song_card = tk.Label(left_panel, text="🎵 No song playing", font=("Arial", 16, "bold"),
                    bg="#34495e", fg="white", width=30, height=2, relief="raised", bd=2)
song_card.pack(pady=20)

# Volume control display
volume_frame = tk.Frame(left_panel, bg="#2c3e50")
volume_frame.pack(pady=15)

tk.Label(volume_frame, text="Volume:", font=("Arial", 14), 
        bg="#2c3e50", fg="#ecf0f1").pack(side="left")

volume_bar = ttk.Progressbar(volume_frame, orient="horizontal", 
                           length=150, mode="determinate")
volume_bar.pack(side="left", padx=10)
volume_bar['value'] = volume_level

volume_percent = tk.Label(volume_frame, text=f"{volume_level}%", font=("Arial", 12),
                         bg="#2c3e50", fg="#ecf0f1")
volume_percent.pack(side="left")

# Gesture history
history_title = tk.Label(left_panel, text="Recent Gestures:", font=("Arial", 14, "bold"),
                        bg="#2c3e50", fg="#ecf0f1")
history_title.pack(pady=(20, 10))

history_frame = tk.Frame(left_panel, bg="#2c3e50")
history_frame.pack(pady=10)
history_labels = []
for _ in range(5):
    lbl = tk.Label(history_frame, text="⬜", font=("Arial", 24), bg="#2c3e50")
    lbl.pack(side="left", padx=8)
    history_labels.append(lbl)

# Action feedback label
feedback_label = tk.Label(left_panel, text="", font=("Arial", 16, "bold"),
                         bg="#2c3e50", fg="#28a745")
feedback_label.pack(pady=15)

# Status bar
status_var = tk.StringVar(value="Ready - Show your hand to the camera")
status_bar = tk.Label(left_panel, textvariable=status_var, font=("Arial", 10),
                     bg="#1e2a36", fg="#95a5a6", relief="sunken", bd=1, anchor="w")
status_bar.pack(fill="x", side="bottom", pady=(10, 0))

# Right panel - Webcam with modern style
right_panel = tk.Frame(content_frame, bg="#1e2a36")
right_panel.pack(side="right", fill="both", expand=True)

# Webcam container with modern frame
webcam_container = tk.Frame(right_panel, bg="#1e2a36")
webcam_container.pack(pady=20)

# Modern style title
modern_title = tk.Label(webcam_container, text="🤖 AI VISION CAMERA", 
                       font=("Arial", 18, "bold"), fg="#3498db", bg="#1e2a36")
modern_title.pack(pady=(0, 15))

canvas_width, canvas_height = 800, 500
canvas = tk.Label(webcam_container, bg="black", relief="flat")
canvas.pack()

# Instructions
instructions = tk.Label(right_panel, 
                       text="📹 Position hand in view • 🎵 Use natural gestures • 🎭 Check guide for help",
                       font=("Arial", 11), bg="#1e2a36", fg="#bdc3c7", wraplength=600)
instructions.pack(pady=10)

# ----------------------------- Webcam thread -----------------------------
frame_counter = 0

def start_webcam():
    global frame_counter
    cap = cv2.VideoCapture(0)
    seq = []
    gesture_locked = False

    emoji_dict = {
        "play": "✋", "pause": "✊", "next": "✌️",
        "previous": "☝️", "volume_up": "👍", "volume_down": "👎"
    }

    def update_frame():
        global song_index, current_gesture, frame_counter
        nonlocal seq, gesture_locked

        ret, frame = cap.read()
        if not ret:
            root.after(10, update_frame)
            return
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        gesture = "None"
        confidence = 0

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                landmarks = [[lm.x, lm.y, lm.z] for lm in handLms.landmark]
                seq.append(landmarks)
        else:
            seq = []
            gesture_locked = False

        if len(seq) > SEQ_LENGTH:
            seq.pop(0)

        if len(seq) == SEQ_LENGTH:
            x = preprocess_sequence(seq)
            y = model.predict(x, verbose=0)[0]
            gid = np.argmax(y)
            confidence = np.max(y)
            gesture = LABELS[gid] if confidence > CONF_THRESHOLD else "Collecting..."
            
            # Update gesture display with action names
            if gesture != current_gesture:
                current_gesture = gesture
                if gesture in gesture_display_map:
                    emoji, action_name, color = gesture_display_map[gesture]
                    gesture_display.config(text=emoji)
                    gesture_label.config(text=f"{action_name}")
                    gesture_label.config(fg=color)
                else:
                    gesture_display.config(text="⬜")
                    gesture_label.config(text="Show your hand", fg="#bdc3c7")

            confidence_var.set(confidence*100)
            confidence_text.config(text=f"Accuracy: {confidence*100:.1f}%")

            # Update history
            if gesture not in ["Collecting...", "None"]:
                gesture_history.appendleft(gesture)
                for i, lbl in enumerate(history_labels):
                    if i < len(gesture_history):
                        hist_gesture = gesture_history[i]
                        if hist_gesture in gesture_display_map:
                            lbl.config(text=gesture_display_map[hist_gesture][0])
                        else:
                            lbl.config(text="⬜")
                    else:
                        lbl.config(text="⬜")

            # Perform actions
            if gesture in ["play", "pause", "next", "previous"] and not gesture_locked:
                perform_action(gesture)
                gesture_locked = True
            elif gesture in ["volume_up", "volume_down"] and confidence > CONF_THRESHOLD:
                perform_action(gesture)

            if gesture_locked and gesture not in ["play", "pause", "next", "previous"]:
                gesture_locked = False

        # Update song card
        if song_list:
            song_name = song_list[song_index].replace('.mp3', '')
            song_card.config(text=f"🎵 {song_name}")
            
            # Add visual indicator for playing state
            if pygame.mixer.music.get_busy():
                song_card.config(bg="#27ae60")  # Green when playing
            else:
                song_card.config(bg="#e74c3c")  # Red when paused

        # Apply modern style frame to webcam feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (canvas_width, canvas_height))
        
        # Add modern effects
        frame_with_effects = create_modern_frame(frame_rgb, frame_counter)
        frame_counter += 1
        
        im_pil = Image.fromarray(frame_with_effects)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        canvas.imgtk = imgtk
        canvas.configure(image=imgtk)

        root.after(10, update_frame)

    update_frame()

# Start the application
update_volume_display()
update_status("Application started - Ready for gesture control")
threading.Thread(target=start_webcam, daemon=True).start()
update_visualization()  # Start animation loop

root.mainloop()