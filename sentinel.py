import face_recognition
import cv2
import os
import pyttsx3
import time
import speech_recognition as sr

class SentinelAgent:
    """
    An AI guard agent that monitors a room using a webcam, microphone, and speakers.
    """
    def __init__(self, trusted_faces_dir='trusted_faces'):
        # 1. Initialize Text-to-Speech (TTS) Engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150) # Speaking rate

        # 2. Initialize Speech Recognition
        self.recognizer = sr.Recognizer()

        # 3. Load trusted faces from the directory
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_trusted_faces(trusted_faces_dir)

        # 4. Intruder detection state variables
        self.intruder_detected = False
        self.last_warning_time = None
        self.escalation_level = 0
        self.unrecognized_face_start_time = None
        
        # A grace period before flagging an unknown person as an intruder
        self.GRACE_PERIOD_SECONDS = 3 

    def speak(self, text):
        """Converts text to speech."""
        print(f"SENTINEL: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

    def load_trusted_faces(self, directory):
        """Loads and encodes faces from the trusted_faces directory."""
        print("Loading trusted faces...")
        if not os.path.exists(directory):
            print(f"Error: Directory '{directory}' not found. Please create it and add images.")
            exit()

        for filename in os.listdir(directory):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(directory, filename)
                name = os.path.splitext(filename)[0].replace("_", " ")
                try:
                    image = face_recognition.load_image_file(path)
                    # Use the first face found in the image
                    encoding = face_recognition.face_encodings(image)[0]
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                    print(f" - Loaded {name}")
                except (IndexError, FileNotFoundError):
                    print(f"Warning: Could not process or find a face in {filename}.")
        print("Trusted faces loaded successfully.")

    def handle_intruder_escalation(self, frame):
        """Manages the escalating verbal warnings."""
        current_time = time.time()

        # First time an intruder is confirmed after the grace period
        if self.escalation_level == 0:
            self.speak("Hello? Can I help you? My owner isn't here right now.")
            self.escalation_level = 1
            self.last_warning_time = current_time
            return

        # Subsequent warnings
        time_since_last_warning = current_time - self.last_warning_time
        
        if self.escalation_level == 1 and time_since_last_warning > 10:
            self.speak("You don't seem to be on the trusted list. This area is being monitored.")
            self.escalation_level = 2
            self.last_warning_time = current_time
        elif self.escalation_level == 2 and time_since_last_warning > 10:
            self.speak("You are not authorized to be here. Your presence is being recorded. Please leave immediately.")
            # Save a snapshot of the intruder
            snapshot_path = f"intruder_{int(current_time)}.jpg"
            cv2.imwrite(snapshot_path, frame)
            print(f"Intruder snapshot saved to {snapshot_path}")
            self.escalation_level = 3
            self.last_warning_time = current_time
        elif self.escalation_level == 3 and time_since_last_warning > 5:
            self.speak("Warning! Intrusion detected. If you do not vacate the premises, I will have to take further action.")
            self.escalation_level = 4 # Max level
            self.last_warning_time = current_time

    def reset_intruder_state(self):
        """Resets all intruder-related flags."""
        if self.intruder_detected:
            print("Threat neutralized or trusted person identified. Resetting state.")
        self.intruder_detected = False
        self.unrecognized_face_start_time = None
        self.escalation_level = 0
        self.last_warning_time = None

    def start_guard_mode(self):
        """Activates the webcam and begins monitoring."""
        self.speak("Guard mode activated. I am now monitoring the room.")
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            self.speak("Error: Could not access the webcam.")
            return

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Process only every other frame to save resources
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = small_frame[:, :, ::-1] # Convert BGR to RGB

            # Find all faces in the current frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            is_trusted_person_present = False
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.6)
                if True in matches:
                    is_trusted_person_present = True
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                    print(f"Recognized: {name}")
                    self.reset_intruder_state()
                    break # Stop checking once one trusted person is found

            # Intrusion detection logic
            if not is_trusted_person_present and face_encodings:
                if self.unrecognized_face_start_time is None:
                    # Start the timer when an unknown face first appears
                    self.unrecognized_face_start_time = time.time()
                elif time.time() - self.unrecognized_face_start_time > self.GRACE_PERIOD_SECONDS:
                    # If unknown face persists beyond the grace period, it's an intruder
                    self.intruder_detected = True
                    self.handle_intruder_escalation(frame)
            else:
                # If no faces are detected or a trusted person is present, reset the timer
                self.unrecognized_face_start_time = None
                if is_trusted_person_present:
                     self.reset_intruder_state()

            # Display the video feed (optional, for monitoring/debugging)
            cv2.imshow('Sentinel Guard View', frame)

            # Press 'q' to quit the guard mode
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
        self.speak("Guard mode deactivated.")

    def listen_for_command(self):
        """Listens for a spoken command from the microphone."""
        with sr.Microphone() as source:
            print("\nListening for a command...")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=4)
                command = self.recognizer.recognize_google(audio).lower()
                print(f"You said: '{command}'")
                return command
            except sr.WaitTimeoutError:
                return ""
            except sr.UnknownValueError:
                return "" # Could not understand audio
            except sr.RequestError:
                print("Could not connect to speech service.")
                return ""

    def run(self):
        """The main loop for the agent."""
        # Using your saved information to personalize the greeting
        self.speak("Hello [Your Name]. Sentinel agent is ready. Say 'guard my room' to activate or 'exit' to shut down.")
        while True:
            command = self.listen_for_command()
            if "guard my room" in command:
                self.start_guard_mode()
                self.speak("I am ready for the next command.")
            elif "exit" in command or "stop" in command:
                self.speak("Shutting down. Goodbye!")
                break

if __name__ == '__main__':
    agent = SentinelAgent()
    agent.run()
