import speech_recognition as sr
import time

class VoiceController:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()

        self.current_mode = "VR ACTIVE"
        self.last_switch_time = 0
        self.cooldown = 2.0  # seconds

        self.commands = {
            "volume": "VOLUME",
            "brightness": "BRIGHTNESS",
            "mouse": "CURSOR",
            "cursor": "CURSOR",
            "stop": "NONE"
        }

        self.active = False
        self.last_command_detected = False   # ✅ CORRECT FLAG

    def listen(self):
        while self.active:
            try:
                with self.mic as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                    audio = self.recognizer.listen(source)

                text = self.recognizer.recognize_google(audio).lower()
                print("[VOICE]:", text)

                now = time.time()
                if now - self.last_switch_time < self.cooldown:
                    continue

                for word, mode in self.commands.items():
                    if word in text:
                        self.current_mode = mode
                        self.last_switch_time = now
                        self.last_command_detected = True   # ✅ EVENT FLAG
                        print(f"[MODE CHANGED → {mode}]")

                        if mode == "NONE":
                            self.active = False

                        break

            except sr.UnknownValueError:
                pass
            except Exception as e:
                print("Voice error:", e)
