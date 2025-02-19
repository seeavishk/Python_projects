import cv2
import numpy as np
from typing import Tuple, List
import tkinter as tk
from tkinter import messagebox  #bss sundarta ke liye...
import threading
import time
from collections import deque
#Defined the manager in here jisko import krunga in the main.py

class VisualizationManager:
    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.warning_alpha = 0.0
        self.last_popup_time = 0
        self.popup_cooldown = 4 
        
        
        self.root = tk.Tk()
        self.root.withdraw()  # hide the main window
        
        # For trend analysis
        self.probability_history = deque(maxlen=30)  # Store last 30 frames
        self.trend_threshold = 1.5  #5 ke saath bhi try krunga..
        self.predictive_warning = False
        self.last_predictive_popup = 0
        
        # Add camera disruption detection
        self.last_frame_time = time.time()
        self.frame_timeout = 0.5  # 500ms threshold for missed frames
        self.camera_disrupted = False
        
    def analyze_trend(self, current_probability: float) -> bool:
        """Analyze if probability is trending towards dangerous levels"""
        self.probability_history.append(current_probability)
        
        if len(self.probability_history) < 10:  # Need at least 10 frames for trend
            return False
            
        # Calculate rate of change over last 10 frames
        recent_values = list(self.probability_history)[-10:]
        rate_of_change = (recent_values[-1] - recent_values[0]) / len(recent_values)
        
        # Predict probability in next 5 seconds (assuming 30 fps)
        predicted_probability = current_probability + (rate_of_change * 150)  # 30fps * 5sec
        
        return (rate_of_change > self.trend_threshold and 
                predicted_probability > 70 and 
                current_probability > 40)  # Current probability must be significant
    
    def check_camera_disruption(self, frame) -> bool:
        """Check if camera feed is disrupted"""
        current_time = time.time()
        
        # Check for sudden motion blur or darkness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        
        # Update last frame time
        time_gap = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Detect disruption conditions
        is_disrupted = (
            time_gap > self.frame_timeout or  # Frame delay
            blur_score < 50 or                # Sudden blur
            brightness < 30 or                # Sudden darkness
            brightness > 250                  # Sudden brightness
        )
        
        if is_disrupted and not self.camera_disrupted:
            self.show_popup_warning(100, camera_alert=True)
            self.camera_disrupted = True
        elif not is_disrupted:
            self.camera_disrupted = False
            
        return is_disrupted
    
    def show_popup_warning(self, probability: float, predictive: bool = False, 
                          weapon_alert: bool = False, camera_alert: bool = False):
        """Show a popup warning in a separate thread"""
        current_time = time.time()
        if camera_alert:
            if current_time - self.last_popup_time >= self.popup_cooldown:
                self.last_popup_time = current_time
                thread = threading.Thread(
                    target=self._display_camera_alert
                )
                thread.daemon = True
                thread.start()
        elif predictive:
            if current_time - self.last_predictive_popup >= self.popup_cooldown:
                self.last_predictive_popup = current_time
                thread = threading.Thread(
                    target=self._display_predictive_popup,
                    args=(probability,)
                )
                thread.daemon = True
                thread.start()
        else:
            if current_time - self.last_popup_time >= self.popup_cooldown:
                self.last_popup_time = current_time
                thread = threading.Thread(
                    target=self._display_popup,
                    args=(probability,)
                )
                thread.daemon = True
                thread.start()
    
    def _display_popup(self, probability: float):
        """Display shorter popup warning at bottom"""
        messagebox.showwarning(
            "⚠️",  # Minimal title
            f"Risk Level: {probability:.1f}%\n"
            f"Take Action Now!"
        )
    
    def _display_camera_alert(self):
        """Display camera disruption alert"""
        messagebox.showwarning(
            "🎥",
            "Camera Disruption Detected!\n"
            "Possible Security Threat"
        )
    
    def _display_predictive_popup(self, probability: float):
        """Display predictive warning popup"""
        messagebox.showwarning(
            "🚨 Alert",  # Shorter title
            f"Risk Increasing!\n"
            f"Level: {probability:.1f}%\n"
            f"• Act now\n"
            f"• Control crowd\n"
            f"• Ready protocols"
        )
        
    def draw_boxes(self, frame, boxes, color=(0, 255, 0)):
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame
    
    def draw_stats(self, frame, result):
        # Check for camera disruption
        camera_disrupted = self.check_camera_disruption(frame)
        
        # Determine panel color based on conditions
        if camera_disrupted:
            panel_color = (0, 0, 255)  # Red for camera disruption
        elif result.new_weapon_detected or result.stampede_probability > 70:
            panel_color = (0, 0, 255)  # Red for danger
        else:
            panel_color = (0, 255, 0)  # Green for normal
            
        # Draw statistics panel at bottom of frame
        panel_height = 100
        panel_y = self.frame_height - panel_height - 10
        
        cv2.rectangle(frame, 
                     (10, panel_y), 
                     (300, self.frame_height - 10), 
                     panel_color, -1)
        cv2.rectangle(frame, 
                     (10, panel_y), 
                     (300, self.frame_height - 10), 
                     (255, 255, 255), 1)
        
        # Draw stats with smaller font at bottom
        font_size = 0.4  # Even smaller font
        base_y = panel_y + 25
        line_spacing = 20
        
        cv2.putText(frame, f"People: {result.people_count}", 
                   (20, base_y), self.font, font_size, (255, 255, 255), 1)
        cv2.putText(frame, f"Speed: {result.movement_speed:.1f}", 
                   (20, base_y + line_spacing), self.font, font_size, (255, 255, 255), 1)
        cv2.putText(frame, f"Panic: {result.panic_expressions}", 
                   (20, base_y + 2 * line_spacing), self.font, font_size, (255, 255, 255), 1)
        
        # Risk level with dynamic color
        risk_text = f"Risk: {result.stampede_probability:.1f}%"
        if camera_disrupted:
            risk_text = "CAMERA DISRUPTED - SECURITY ALERT"
            
        cv2.putText(frame, risk_text, 
                   (20, base_y + 3 * line_spacing), self.font, font_size, 
                   (0, 0, 255) if camera_disrupted or result.stampede_probability > 70 else (255, 255, 255), 1)
        
        # Show alerts
        if camera_disrupted:
            overlay = frame.copy()
            cv2.putText(overlay, "⚠️ CAMERA DISRUPTION DETECTED!", 
                       (int(self.frame_width/2 - 150), 50),
                       self.font, 1.0, (0, 0, 255), 2)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Show weapon alert if new weapon detected
        if result.new_weapon_detected:
            self.show_popup_warning(result.stampede_probability, weapon_alert=True)
            
            # Add red flashing warning for weapon
            overlay = frame.copy()
            cv2.putText(overlay, "⚠️ WEAPON DETECTED!", 
                       (int(self.frame_width/2 - 150), 50),  # Adjusted position
                       self.font, 1.0, (0, 0, 255), 2)  # Smaller font
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Analyze trend and show predictive warning if needed
        if self.analyze_trend(result.stampede_probability):
            self.predictive_warning = True
            self.show_popup_warning(result.stampede_probability, predictive=True)
        else:
            self.predictive_warning = False
        
        # Draw warning if high risk
        if result.stampede_probability > 70:
            self.warning_alpha = min(1.0, self.warning_alpha + 0.1)
            self.show_popup_warning(result.stampede_probability)
        else:
            self.warning_alpha = max(0.0, self.warning_alpha - 0.1)
            
        if self.warning_alpha > 0 or self.predictive_warning:
            overlay = frame.copy()
            # Add red border around frame
            cv2.rectangle(overlay, (0, 0), 
                        (self.frame_width-1, self.frame_height-1), 
                        (0, 0, 255), 10)
            
            if self.predictive_warning:
                # Add predictive warning text
                cv2.putText(overlay, "WARNING: STAMPEDE RISK INCREASING!", 
                           (int(self.frame_width/2 - 250), int(self.frame_height/2 - 30)),
                           self.font, 1.2, (0, 0, 255), 2)
                cv2.putText(overlay, "PREDICTED HIGH RISK WITHIN 5 SECONDS", 
                           (int(self.frame_width/2 - 280), int(self.frame_height/2 + 30)),
                           self.font, 1.2, (0, 0, 255), 2)
            else:
                # Add current warning text
                cv2.putText(overlay, "WARNING: HIGH STAMPEDE RISK!", 
                           (int(self.frame_width/2 - 200), int(self.frame_height/2)),
                           self.font, 1.5, (0, 0, 255), 2)
                
            alpha = max(self.warning_alpha, 0.7 if self.predictive_warning else 0)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return frame 