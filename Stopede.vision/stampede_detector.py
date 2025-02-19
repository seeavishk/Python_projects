import cv2
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import datetime
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd

@dataclass
class DetectionResult:
    timestamp: datetime.datetime
    people_count: int
    movement_speed: float
    panic_expressions: int
    weapons_detected: List[str]
    stampede_probability: float
    new_weapon_detected: bool = False
    
class StampedeDetector:
    def __init__(self, 
                 people_threshold: int = 25,
                 speed_threshold: float = 40.0,
                 panic_threshold: float = 0.6):
        # Initialize models
        self.yolo_model = YOLO('yolov8n.pt')
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.prev_frame = None
        self.people_threshold = people_threshold
        self.speed_threshold = speed_threshold
        self.panic_threshold = panic_threshold
        
        # Define dangerous objects to detect
        self.dangerous_objects = [
            28,  # umbrella
            43,  # knife
           
            67,  # cell phone
            73,  # laptop
            76,  # scissors
            77,  # teddy bear
            85,  # clock
            86,  # vase
            87,  # pen
            88,  # pencil
            89,  # ruler
            90   # sharp objects
        ]
        
        logging.basicConfig(
            filename='stampede_events.log', #storing of the log of the signal where the harm is being detected..
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Initialize results storage
        self.results_df = pd.DataFrame(columns=[
            'timestamp', 'people_count', 'movement_speed',
            'panic_expressions', 'weapons_detected', 'stampede_probability'
        ])
        
        # JO detect hua tha previously usko use krna to store in the log
        
        self.prev_weapons = set()

    def detect_people(self, frame):
        results = self.yolo_model(frame, classes=[0])  # 0 is the class for person
        return len(results[0].boxes), results[0].boxes

    def analyze_movement(self, frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return 0.0
        
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, current_frame, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        speed = np.mean(np.abs(flow))
        self.prev_frame = current_frame
        return speed

    def detect_panic(self, frame):
        try:
            # Yaha prr normal code hai to detect the face #resource---cvzone
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
    
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            panic_count = 0
            for (x, y, w, h) in faces:
                # Extract face region with some margin
                face_img = frame[max(0, y-20):min(frame.shape[0], y+h+20),
                               max(0, x-20):min(frame.shape[1], x+w+20)]
                
                if face_img.size == 0:  
                    continue
                
                # Yaha se face ko analyze krega ..deepface documnetaional
                result = DeepFace.analyze(
                    face_img,
                    actions=['emotion'],
                    enforce_detection=False,  #pehle false hi rkha gya..
                    detector_backend='opencv'
                )
                
                # Checking the emotions
                if isinstance(result, list):
                    result = result[0]
                #three params for the detection of the face expression
                if result['dominant_emotion'] in ['fear', 'angry', 'surprise']:
                    panic_count += 1
                    
            return panic_count
            
        except Exception as e:
            logging.error(f"Face analysis error: {str(e)}")
            return 0

    def detect_weapons(self, frame):
        # Detect all specified dangerous objects
        results = self.yolo_model(
            frame, 
            classes=self.dangerous_objects,
            conf=0.35  #confidence of the dectection
        )
        
        detected_objects = []
        current_weapons = set()
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf > 0.35:
                object_name = self.yolo_model.names[class_id]
                detected_objects.append(f"{object_name} ({conf:.2f})")
                current_weapons.add(object_name)
        
        # Check for newly introduced weapons
        new_weapons = current_weapons - self.prev_weapons
        self.prev_weapons = current_weapons
        
        return detected_objects, bool(new_weapons)

    def calculate_stampede_probability(self, 
                                    people_count: int,
                                    movement_speed: float,
                                    panic_count: int,
                                    weapons: List[str]) -> float:
        # Enhanced weighted probability calculation
        probability = 0.0
        
        # People count factor (40% weight)
        if people_count > self.people_threshold:
            probability += 0.40 * min(people_count / self.people_threshold, 1.0)
            
        # Movement speed factor (35% weight)
        if movement_speed > self.speed_threshold:
            probability += 0.25 * min(movement_speed / self.speed_threshold, 1.0)
            
        # Panic expression factor (15% weight)
        if panic_count > 0:
            probability += 0.25 * min(panic_count / (people_count or 1), 1.0)
            
        # Dangerous objects factor (10% weight)
        if weapons:
            probability += 0.10 * min(len(weapons) / 2, 1.0)  # Cap at 2 objects
            
        return min(probability, 1.0) * 100

    def process_frame(self, frame):
        # Process frame with all detectors
        people_count, boxes = self.detect_people(frame)
        movement_speed = self.analyze_movement(frame)
        panic_count = self.detect_panic(frame)
        weapons, new_weapon_detected = self.detect_weapons(frame)
        
        # Calculate probability
        probability = self.calculate_stampede_probability(
            people_count, movement_speed, panic_count, weapons
        )
        
        # Create result object
        result = DetectionResult(
            timestamp=datetime.datetime.now(),
            people_count=people_count,
            movement_speed=movement_speed,
            panic_expressions=panic_count,
            weapons_detected=weapons,
            stampede_probability=probability,
            new_weapon_detected=new_weapon_detected
        )
        
        # Log if high risk or new weapon
        if probability > 70 or new_weapon_detected:
            logging.warning(
                f"{'NEW WEAPON DETECTED! ' if new_weapon_detected else ''}"
                f"High stampede risk detected! "
                f"Probability: {probability:.1f}% "
                f"People: {people_count} "
                f"Speed: {movement_speed:.2f} "
                f"Panic: {panic_count} "
                f"Objects: {', '.join(weapons)}"
            )
            
        # Store results
        self.results_df.loc[len(self.results_df)] = [
            result.timestamp, result.people_count,
            result.movement_speed, result.panic_expressions,
            result.weapons_detected, result.stampede_probability
        ]
        
        return result, boxes 