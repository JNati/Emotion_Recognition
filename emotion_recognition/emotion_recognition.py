import os
from datetime import datetime
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from tkinter import messagebox

class EmotionAnalyzer:
    def __init__(self, root, video_frame, video_path=None, on_close_callback=None):
        self.root = root
        self.video_frame = video_frame
        self.video_path = video_path
        self.on_close_callback = on_close_callback
        self.user_data_collected = [False]  # Status for two people
        self.emotion_labels = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
        self.emotion_history = {label: [] for label in self.emotion_labels}
        self.cap = None
        self.running = True
        self.user_dirs = {}
        self.frame = None


        # Load the model
        try:
            self.model = load_model('models/densenet_model.h5')
        except Exception as e:
            print(f"Error loading model: {e}")

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def stop_detection(self):
        self.running = False
        if hasattr(self, 'after_id'):
            self.root.after_cancel(self.after_id)

    def write_emotions_to_file(self, emotions, name):
        video_filename = os.path.basename(self.video_path).split('.')[0]
        current_date = datetime.now().strftime("%Y%m%d")
        user_dir = f"results/emotion_history/{name}_{current_date}"
        self.user_dirs[name] = user_dir 
        #print(self.user_dirs)
        if not os.path.exists(user_dir):
            os.makedirs(user_dir)

        filename = os.path.join(user_dir, f"{video_filename}_emotions.txt")
        with open(filename, "a") as file:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"Time: {current_time}, Name: {name}\n")
            for emotion, probability in emotions.items():
                file.write(f"{emotion}: {probability:.2f}%\n")
            file.write("\n")

    def run_emotion_detection(self):
        person_data = {'name': os.path.splitext(os.path.basename(self.video_path))[0]}

        while self.cap.isOpened() and self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.current_frame = frame

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face_roi = gray_frame[y:y + h, x:x + w]
                face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                resized_face = cv2.resize(face_rgb, (48, 48))
                normalized_face = resized_face / 255.0
                processed_face = np.expand_dims(np.expand_dims(normalized_face, axis=-1), axis=0)
                predictions = self.model.predict(processed_face)
                predicted_emotion_index = np.argmax(predictions)
                predicted_emotion = self.emotion_labels[predicted_emotion_index]
                probability = predictions[0][predicted_emotion_index] * 100
                emotions = {self.emotion_labels[j]: predictions[0][j] * 100 for j in range(len(self.emotion_labels))}
                self.write_emotions_to_file(emotions, person_data['name'])

                for emotion, value in emotions.items():
                    self.emotion_history[emotion].append(value)

            self.update_ui(frame)

        self.root.after(0, self.on_close_callback)
        self.cap.release()
        cv2.destroyAllWindows()

    def start_detection(self):
        if self.video_path is None:
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.video_path)

            if not self.cap.isOpened():
                print(f"Error opening video file: {self.video_path}")
                return

            name = os.path.splitext(os.path.basename(self.video_path))[0]
            current_date = datetime.now().strftime("%Y%m%d")
            user_dir = f"results/emotion_history/{name}_{current_date}"

            if not os.path.exists(user_dir):
                os.makedirs(user_dir)

            self.run_emotion_detection()

    def update_ui(self, frame):
        if not self.root.winfo_exists() or not self.running:
            if callable(self.on_close_callback):
                self.on_close_callback()
            return

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (320, 180))

        img_pil = Image.fromarray(img)

        def update():
            if not self.root.winfo_exists() or not self.running:
                return
            img_tk = ImageTk.PhotoImage(img_pil)
            if self.video_frame.winfo_exists():
                self.video_frame.config(image=img_tk)
                self.video_frame.image = img_tk

        self.after_id = self.root.after(100, update)

class EmotionStatistics:
    def __init__(self, emotion_history, axs, canvas, root, fig):
        self.emotion_history = emotion_history
        self.axs = axs
        self.canvas = canvas
        self.root = root
        self.fig = fig
        self.running = True

    def run_statistics(self):
        if not self.running:
            return

        if not self.root.winfo_exists():
            self.running = False
            return

        for ax in self.axs.flat:
            ax.clear()

        if self.emotion_history:
            latest_values = {emotion: history[-1] for emotion, history in self.emotion_history.items() if history}
            self.axs[0, 0].pie(latest_values.values(), labels=latest_values.keys(), autopct='%1.1f%%', startangle=90)
            self.axs[0, 0].axis('equal')
            self.axs[0, 0].set_title("Aktuális érzelem megoszlás")

            emotion_values = [history for history in self.emotion_history.values() if history]
            if emotion_values:
                self.axs[1, 1].boxplot(emotion_values, tick_labels=[label for label in self.emotion_history.keys()])
                self.axs[1, 1].set_title("Érzelemszázalékok boxplotonként")

        for emotion, values in self.emotion_history.items():
            if values:
                self.axs[1, 0].bar(emotion, np.mean(values))  # Átlagértékek

                smoothed_values = np.convolve(values, np.ones(5) / 5, mode='valid')
                line, = self.axs[0, 1].plot(smoothed_values, label=f"Átlag {emotion}")
                self.axs[0, 1].text(len(smoothed_values) - 1, smoothed_values[-1], emotion,
                                    color=line.get_color(), fontsize=10, va='center')

        self.axs[0, 1].set_title("Mozgóátlag érzelemszázalékokban időarányosan")
        self.axs[1, 0].set_title("Átlagérzelem százalékok kategóriánként")

        self.canvas.draw()
        self.after_id = self.root.after(100,
                                        self.run_statistics)

    def stop_statistics(self):
        if hasattr(self, 'after_id'):
            self.root.after_cancel(self.after_id)

    def save_statistics(self, user_dir):
        current_date = datetime.now().strftime("%Y%m%d")
        output_dir = user_dir

        try:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            image_path = os.path.join(output_dir, f"statistics_{current_date}.png")
            self.fig.savefig(image_path)
            print(f"Statisztikai ábra mentve ide: {image_path}")
        except Exception as e:
            print(f"Hiba a statisztika mentésekor: {e}")


class EmotionApp:
    def __init__(self, video_path):
        self.root = tk.Tk()
        self.root.title("Érzelem Analízis és Videó Megjelenítő")
        self.root.geometry("1800x1100")
        self.video_frame = tk.Label(self.root)
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.emotion_analyzer = EmotionAnalyzer(self.root, self.video_frame, video_path=video_path, on_close_callback=self.on_closing)
        self.statistics = EmotionStatistics(self.emotion_analyzer.emotion_history, self.axs, self.canvas, self.root, self.fig)

        self.root.bind('<q>', lambda event: self.on_closing())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        threading.Thread(target=self.emotion_analyzer.start_detection, daemon=True).start()
        self.statistics.run_statistics()

        self.root.mainloop()

    def on_closing(self):
        self.emotion_analyzer.running = False
        self.statistics.running = False
        print("Elindul")

        if self.emotion_analyzer.user_dirs:
            latest_user_dir = list(self.emotion_analyzer.user_dirs.values())[-1]
            print(latest_user_dir)
            self.statistics.save_statistics(latest_user_dir)

        self.emotion_analyzer.stop_detection()
        self.statistics.stop_statistics()
        if self.emotion_analyzer.cap is not None:
            self.emotion_analyzer.cap.release()
        print("Leáll")

        cv2.destroyAllWindows()
        self.root.destroy()


if __name__ == "__main__":
    video_paths = [
        #"videos/angry1.mp4",
        #"videos/angry2.mp4",
        #"videos/angry3.mp4",
        #"videos/disgust1.mp4",
        #"videos/disgust2.mp4",
        #"videos/disgust3.mp4",
        #"videos/fear1.mp4",
        "videos/fear2.mp4",
        #"videos/fear3.mp4",
        #"videos/happy1.mp4",
        #"videos/happy2.mp4",
        #"videos/happy3.mp4",
        #"videos/neutral1.mp4",
        #"videos/neutral2.mp4",
        #"videos/neutral3.mp4",
        #"videos/sad1.mp4",
        #"videos/sad2.mp4",
        #"videos/sad3.mp4",
        #"videos/surprise1.mp4",
        #"videos/surprise2.mp4",
        #"videos/surprise3.mp4",
]
    for video_path in video_paths:
        print(f"Feldolgozás: {video_path}")
        app = EmotionApp(video_path)






