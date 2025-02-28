import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import time
from PIL import Image, ImageTk
import numpy as np
from stampede_detector import StampedeDetector
from visualization import VisualizationManager

class StampedeDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stampede Detection System")
        self.root.geometry("1000x700")
        self.root.configure(bg='#1e1e1e')
        
        # Animation frames
        self.animation_frames = []
        self.current_frame = 0
        self.animation_running = True
        
        # Create animation frames (simple example)
        for i in range(60):
            frame = np.zeros((300, 500, 3), dtype=np.uint8)
            for j in range(20):
                x = int((i + j * 10) % 500)
                y = 150 + int(30 * np.sin(i/10 + j))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            self.animation_frames.append(frame)
        
        self.setup_ui()
        
    def setup_ui(self):
        title_label = tk.Label(
            self.root,
            text="Stampede Detection System",
            font=("Arial", 24, "bold"),
            fg='#ffffff',
            bg='#1e1e1e'
        )
        title_label.pack(pady=20)
        
        self.canvas = tk.Canvas(
            self.root,
            width=500,
            height=300,
            bg='#2d2d2d',
            highlightthickness=0
        )
        self.canvas.pack(pady=20)
        
        self.status_label = tk.Label(
            self.root,
            text="System Initializing...",
            font=("Arial", 12),
            fg='#00ff00',
            bg='#1e1e1e'
        )
        self.status_label.pack(pady=10)
        
        self.progress = ttk.Progressbar(
            self.root,
            length=400,
            mode='determinate'
        )
        self.progress.pack(pady=20)
        
        self.start_button = tk.Button(
            self.root,
            text="Start Detection",
            command=self.start_detection,
            font=("Arial", 14),
            bg='#007acc',
            fg='white',
            width=20,
            height=2,
            state='disabled'
        )
        self.start_button.pack(pady=20)
        
        info_frame = tk.Frame(self.root, bg='#1e1e1e')
        info_frame.pack(pady=20, fill='x', padx=50)
        
        features = [
            "👥 Real-time People Detection",
            "⚡ Movement Analysis",
            "😨 Panic Expression Detection",
            "📊 Risk Assessment"
        ]
        
        for feature in features:
            feature_label = tk.Label(
                info_frame,
                text=feature,
                font=("Arial", 12),
                fg='#ffffff',
                bg='#1e1e1e',
                anchor='w'
            )
            feature_label.pack(pady=5)
        
        self.animate()
        self.initialize_system()
        
    def animate(self):
        if self.animation_running:
            frame = self.animation_frames[self.current_frame]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img_tk = ImageTk.PhotoImage(image=img)
            
            self.canvas.delete("all")
            self.canvas.create_image(250, 150, image=img_tk)
            self.canvas.image = img_tk
            
            self.current_frame = (self.current_frame + 1) % len(self.animation_frames)
            self.root.after(50, self.animate)
    
    def initialize_system(self):
        def init_thread():
            steps = [
                "Loading YOLO model...",
                "Initializing face detection...",
                "Setting up visualization...",
                "Calibrating detection parameters...",
                "System ready!"
            ]
            
            for i, step in enumerate(steps):
                time.sleep(1)  
                self.status_label.config(text=step)
                self.progress['value'] = (i + 1) * 20
                
                if i == len(steps) - 1:
                    self.start_button.config(state='normal')
                    self.status_label.config(fg='#00ff00')
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def start_detection(self):
        self.animation_running = False
        self.root.destroy()
        self.start_detection_system()
    
    def start_detection_system(self):
        # ip_camera_url = "http://10.252.181.128:8080/video"  # Replace with your phone's IP camera URL
        # cap = cv2.VideoCapture(ip_camera_url)
        cap = cv2.VideoCapture(0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if not cap.isOpened():
            print("Error: Cannot connect to phone camera. Check IP address and connection.")
            return

        frame_width = 600  # Set a smaller frame width
        frame_height = 600  # Set a smaller frame height

        detector = StampedeDetector()
        visualizer = VisualizationManager(frame_width, frame_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to fetch frame from phone camera.")
                break

            frame = cv2.resize(frame, (frame_width, frame_height))  # Resize the frame
            result, boxes = detector.process_frame(frame)
            frame = visualizer.draw_boxes(frame, boxes)
            frame = visualizer.draw_stats(frame, result)

            cv2.imshow('STOPEDE.VISION', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        detector.results_df.to_csv('stampede_detection_results.csv', index=False)
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = StampedeDetectionGUI()
    app.run()
