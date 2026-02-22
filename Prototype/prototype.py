import numpy as np
import cv2
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import scipy.signal as signal
import sys
import requests
from picamera2 import Picamera2
from multiprocessing import Process,Pipe,Barrier
import requests
import cv2
import io
import soundfile as sf


# --- TELEGRAM CONFIG ---
BOT_TOKEN = "NA"
CHAT_ID = "NA"

# --- CONFIGURATION ---
# Audio
AUDIO_MODEL_PATH = "audio_classifier.tflite"
AUDIO_IMG_SIZE = 128
AUDIO_SR = 16000
HW_SR = 44100
MIC_INDEX = 1  # <--- CHECK THIS (Run 'python3 -m sounddevice' if audio fails)
DURATION = 3.0
AUDIO_LABELS = ["ADULT_SOUND", "CHILD_CRY", "CHILD_SOUND", "NOISE", "PET_SOUND","SILENCE"]
YAMNET_PATH = r'yamnet.tflite'

REC_SAMPLES = int(DURATION * HW_SR)

# Video
VIDEO_MODEL_PATH = "yolon640_float32.tflite" 
VIDEO_IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
VIDEO_LABELS = ["ADULT", "CHILD", "PET"] 
COLORS = [[0, 255, 0], [0, 0, 255], [0, 128, 255]]
MAX_WIN_WIDTH = 1280
MAX_WIN_HEIGHT = 720




def send_telegram_image(cv2_frame, caption=""):
    """
    Sends an OpenCV image (numpy array) directly to Telegram
    without saving it to a file first.
    """
    try:
        # 1. Encode the frame to JPEG bytes in memory
        # success, buffer = cv2.imencode(".jpg", cv2_frame)
        ret, buffer = cv2.imencode('.jpg', cv2_frame)
        
        if not ret:
            print(f"\n❌ Could not encode image!")
            return

        # 2. Convert to a BytesIO object (acts like a file)
        io_buf = io.BytesIO(buffer)
        io_buf.name = 'alert.jpg'  # Telegram needs a filename

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
        data = {"chat_id": CHAT_ID, "caption": caption}
        files = {"photo": io_buf}

        requests.post(url, data=data, files=files, timeout=10)

    except Exception as e:
        print(f"\n❌ Telegram Fail: {e}")
        return


def send_telegram_sound(audio, caption=""):
    """
    Sends an audio directly to Telegram displayed as voice note
    without saving it to a file first.
    """
    try:
        
        ogg_buffer = io.BytesIO()
        sf.write(ogg_buffer, audio, HW_SR, format='OGG', subtype='VORBIS')

        ogg_buffer.seek(0)

        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendVoice"
        data = {"chat_id": CHAT_ID, "caption": caption}
        files = {"voice": ("chunk.ogg", ogg_buffer, "audio/ogg")}

        requests.post(url, data=data, files=files, timeout=10)

    except Exception as e:
        print(f"\n❌ Telegram Fail: {e}")
        return

def letterbox(image, target_size=(480, 480), color=(114, 114, 114)):
    """
    Resizes an image to fit within target_size while maintaining aspect ratio,
    padding the remaining area with a specific color.

    Args:
        image (np.ndarray): Original input image (H, W, C).
        target_size (tuple): Desired output size (width, height).
        color (tuple): Padding color (B, G, R). Default is YOLO gray (114, 114, 114).

    Returns:
        np.ndarray: The letterboxed image ready for inference.
        tuple: (ratio, (padding_width, padding_height)) - Useful for rescaling coords later.
    """
    target_w, target_h = target_size
    h, w = image.shape[:2]

    # 1. Calculate the scaling ratio (min of width or height ratio)
    scale = min(target_w / w, target_h / h)
    
    # 2. Calculate the new unpadded dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 3. Resize the image
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 4. Calculate padding (deltas)
    delta_w = target_w - new_w
    delta_h = target_h - new_h

    # Divide padding by 2 to center the image
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # 5. Add border (padding)
    letterboxed = cv2.copyMakeBorder(
        resized_image, 
        top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=color
    )

    return letterboxed, scale, (left, top)

def draw_detection(canvas,width,height,class_id,detection):

    x1, y1 = int(detection[0]*width), int(detection[1]*height)
    x2, y2 = int(detection[2]*width), int(detection[3]*height)
    color = COLORS[class_id]
    label = VIDEO_LABELS[class_id]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    cv2.putText(canvas, f"{label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return canvas

def eye_process(brain_conn,lock):
    """
    Captures image, Normalizes to Float32, and parses [1, 300, 6] output.
    """
    # --- INITIALIZE CAMERA ---
    print(f"\nEYE : Initializing Picamera2...")
    try:
        picam2 = Picamera2()
        # Configure 640x480 (Close to 480x480 model size)
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        print(f"EYE : ✅ Camera Started.")
    except Exception as e:
        print(f"EYE : ❌ Camera Failed {e}")
        sys.exit(1)

    print(f"EYE : Initializing tflite interpreter")
    try:
        video_interpreter = tflite.Interpreter(model_path=VIDEO_MODEL_PATH)
        video_interpreter.allocate_tensors()
        video_input_details = video_interpreter.get_input_details()
        video_output_details = video_interpreter.get_output_details()
    except Exception as e:
        print(f"EYE : ❌ [AI] Model Load Failed: {e}")
        sys.exit(5)
    

    while True:

        lock.wait()

        print(f"\nEYE : Capturing frame", end=" ", flush=True)

        # 1. Capture
        try:
            frame = picam2.capture_array()
        except Exception as e:
            print(f"EYE : ❌ Capture Error {e}")
            sys.exit(2)

        #Convert bgr to rgb
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Process (Normalize 0-255 -> 0.0-1.0)
        img, ratio, (dw, dh) = letterbox(frame, (VIDEO_IMG_SIZE, VIDEO_IMG_SIZE))
        
        input_data = np.expand_dims((img.astype(np.float32) / 255.0), axis=0)
        
        # 3. Inference
        video_interpreter.set_tensor(video_input_details[0]['index'], input_data)
        video_interpreter.invoke()

        # 4. Parse [1, 300, 6] Output
        # The output is a list of 300 potential boxes.
        # Format usually: [x1, y1, x2, y2, SCORE, CLASS_ID]
        output_data = video_interpreter.get_tensor(video_output_details[0]['index'])[0] 

        detected_objects = []

        
        orig_h, orig_w = img.shape[:2]
        scale = min(MAX_WIN_WIDTH/orig_w, (MAX_WIN_HEIGHT)/orig_h)
        new_w, new_h = int(orig_w*scale), int(orig_h*scale)
        canvas = cv2.resize(img, (new_w, new_h))
        

        # Loop through all 300 rows
        for detection in output_data:
            score = detection[4]   # Index 4 is usually Confidence Score
            class_id = int(detection[5]) # Index 5 is usually Class ID
        
            if score > CONFIDENCE_THRESHOLD:
                # Map ID to Name
                if class_id < len(VIDEO_LABELS):
                    label_name = VIDEO_LABELS[class_id]
                    canvas=draw_detection(canvas,new_w,new_h,class_id,detection)
                    detected_objects.append(label_name)   
        
        # --- INSTANT PRINT ---
        if len(detected_objects) > 0:
            # Remove duplicates for cleaner printing (e.g. "CHILD, CHILD" -> "CHILD")
            unique_objects = list(set(detected_objects))
            print(f"\nEYE : 👉 Found: {', '.join(unique_objects).upper()}")
        else:
            print(f"\nEYE : 👉 Found: NOTHING")
        
        brain_conn.send((detected_objects,canvas))

def ear_process(brain_conn,lock):

    
    print(f"\nEAR : Initializing tflite interpreters")


    try:
        audio_interpreter = tflite.Interpreter(model_path=AUDIO_MODEL_PATH)
        audio_interpreter.allocate_tensors()
        input_idx = audio_interpreter.get_input_details()[0]['index']
        output_idx = audio_interpreter.get_output_details()[0]['index']
        
        yamnet_interpreter = tflite.Interpreter(model_path=YAMNET_PATH)
        yamnet_interpreter.allocate_tensors()
        yamnet_input_idx = yamnet_interpreter.get_input_details()[0]['index']
        yamnet_output_idx = yamnet_interpreter.get_output_details()[1]['index']
    except Exception as e:
        print(f"\nEAR : ❌ [AI] Model Load Failed: {e}")
        sys.exit(6)


    while True:

        lock.wait()

        print(f"\nEAR: Listening ({DURATION})...", end=" ", flush=True)
    
        try:
            
            audio = sd.rec(int(DURATION * HW_SR), samplerate=HW_SR, channels=1, device=MIC_INDEX)
            sd.wait()
        except Exception as e:
            print(f"\nEAR : ❌ MIC ERROR {e}")
            sys.exit(3)
        
        y_hw = audio.flatten()

        # Resample from HW_SR (e.g. 44100) to 16000
        y_16k = signal.resample_poly(y_hw, AUDIO_SR, HW_SR)

        y_16k = y_16k.astype(np.float32)

        yamnet_interpreter.resize_tensor_input(yamnet_input_idx, [len(y_16k)],strict=True)
        yamnet_interpreter.allocate_tensors() # Re-allocate for new shape

        yamnet_interpreter.set_tensor(yamnet_input_idx, y_16k)
        yamnet_interpreter.invoke()

        embeddings = yamnet_interpreter.get_tensor(yamnet_output_idx)
            
        # 2. Global Average Pooling (Reduce 5s of features to 1 vector)
        # embeddings shape is (N, 1024) -> becomes (1024,)
        global_embedding = np.mean(embeddings, axis=0)

        input_data = np.expand_dims(global_embedding, axis=0)
        
        audio_interpreter.set_tensor(input_idx, input_data)
        audio_interpreter.invoke()
        prediction = audio_interpreter.get_tensor(output_idx)
    
        class_idx = np.argmax(prediction)
        label = AUDIO_LABELS[class_idx]
        confidence = prediction[0][class_idx] * 100

        
        detections=[]
        msg_audio=0

        if confidence > 0.60:
            detections.append(label)
            msg_audio=audio
            print(f'EAR : Found {label}')

        if len(detections)==0:
            print(f'\nEAR : Found Nothing')

        brain_conn.send((detections,msg_audio))

def evaluate_state(eye_detections,ear_detections):
    """
    Docstring for evaluate_state
    
    :param eye_detections: object containing the detections from the eye
    :param ear_detections: object containing the detections from the ear

    return a tuple (num,entity)

    the num indicates if safety state
        0: Safe
        1: Risk
        2: Critical
    
    entity indicates , if something is detected, what has been detected
        0 : ADULT
        1 : CHILD
        2 : PET
        3 : NONE

    det_types
       0 : nothing
       1 : image
       2 : sound
    """
    #Evaluate eye data

    childDetected = 'CHILD' in eye_detections
    petDetected = 'PET' in eye_detections
    adultDetected = 'ADULT' in eye_detections
    adultHeard = 'ADULT_SOUND' in ear_detections

    if adultDetected or adultHeard :
        print("✅ Safe Adult detected")
        return (0,0,0)
    if childDetected :
        print("⚠️ WARNING: Unaccompained Child detected (Image)")
        return (1,1,1)
    if petDetected :
        print("⚠️ WARNING: Unaccompained Pet detected (Image)")
        return (1,2,1)
    
    #Evaluate ear data
    petHeard = 'PET_SOUND' in ear_detections
    childHeard = 'CHILD_SOUND' in ear_detections
    childCry = 'CHILD_CRY' in ear_detections
    noiseHeard = 'NOISE' in ear_detections
    silenceHeard = 'SILENCE' in ear_detections

    if childCry or childHeard:
        print(f"\n⚠️ WARNING: Child CSounds detected")
        return (1,1,2)
    if petHeard :
        print(f"\n⚠️ WARNING: Pet sounds detected")
        return (1,2,2)
    if noiseHeard :
        print(f"\n Noise Nothing detected")
        return (0,3,2)
    if silenceHeard :
        print(f"\n✅ Silence detected")
        return (0,3,2)
    
    print(f"\n✅ Safe Nothing detected")
    
    return (0,0,0)

def monitor(eye_conn,ear_conn,lock):
    
    try:
        while True:

            # WAIT FOR DATA FROM EAR AND EYE
            lock.wait()
            # 1. GET EYE DATA
            eye_detections,canvas = eye_conn.recv()
            # 2. AUDIO
            ear_detections,audio = ear_conn.recv()
            
            safety_level,entity_detected,dec_type = evaluate_state(eye_detections,ear_detections)
            
            if safety_level == 1:
                if dec_type == 1:
                    send_telegram_image(canvas,f"🚨 WARNING: {VIDEO_LABELS[entity_detected]} alone inside the car!")
                else:
                    send_telegram_sound(audio,f"🚨 WARNING: {VIDEO_LABELS[entity_detected]} alone inside the car!")

    except KeyboardInterrupt:
        sys.exit(0)
    except (EOFError, BrokenPipeError):
        print("One of the process stopped working correctly")
        exit()

if __name__ == "__main__":

    brain_to_eye_conn, eye_to_brain_conn = Pipe()

    brain_to_ear_conn, ear_to_brain_conn = Pipe()

    lock = Barrier(3)

    eye = Process(target=eye_process,args=(eye_to_brain_conn,lock))
    ear = Process(target=ear_process,args=(ear_to_brain_conn,lock))

    eye.deamon=True
    ear.deamon=True

    eye.start()
    ear.start()

    monitor(brain_to_eye_conn,brain_to_ear_conn,lock)
