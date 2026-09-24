import socket
import requests
import numpy as np
import sounddevice as sd
import tensorflow as tf
import tensorflow_hub as hub


YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
ATOMIZER_METHOD = "http" # Valid labels: http or udp
ATOMIZER_IP_ADDRESS = "192.168.1.45"
ATOMIZER_PORT = 80
SAMPLE_RATE = 16000  # YAMNet expects audio at 16 kHz
BLOCK_SIZE = 16000  # Grab 1-second chunks of audio for processing
VB_CABLE = None


def load_model():
    '''Load the YAMNet model from TensorFlow Hub.'''
    print("Loading YAMNet model...")
    model = hub.load(YAMNET_MODEL_URL)
    return model

def load_labels(model):
    '''Load the class labels for YAMNet.'''
    print("Loading class labels...")
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    labels = []
    with open(class_map_path, "r") as f:
        labels = [line.split(",")[2].strip().strip('"') for line in f.readlines()[1:]]
    return labels

def trigger_atomizer_http(slot:int, duration:float=2.0):
    '''Trigger the atomizer to release a scent for a specified duration.'''
    print(f"Triggering atomizer slot: {slot} for {duration} seconds.")
    url = f"http://{ATOMIZER_IP_ADDRESS}:{ATOMIZER_PORT}/api/trigger"
    payload = {"slot": slot, "duration": duration}
    try:
        response = requests.post(url, json=payload, timeout=2)
        if response.status_code == 200:
            print("Atomizer triggered successfully.")
    except requests.RequestException as e:
        print(f"Error triggering atomizer: {e}")

def trigger_atomizer_raw_udp(slot:int, duration:float=1.0):
    '''Trigger the atomizer to release a scent for a specified duration using raw UDP.'''
    print(f"Triggering atomizer slot: {slot} for {duration} seconds via UDP.")
    scent_commands = {
        1: b'\x01\x00\x00\x00',  # Example command for slot 1
        2: b'\x02\x00\x00\x00',  # Example command for slot 2
        3: b'\x03\x00\x00\x00',  # Example command for slot 3
        4: b'\x04\x00\x00\x00',  # Example command for slot 4
        5: b'\x05\x00\x00\x00',  # Example command for slot 5
        6: b'\x06\x00\x00\x00',  # Example command for slot 6
    }
    payload = scent_commands.get(slot)
    
    if not payload:
        return False

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        sock.sendto(payload, (ATOMIZER_IP_ADDRESS, ATOMIZER_PORT))
        return True
    except Exception:
        return False
    finally:
        sock.close()

def clean_air():
    '''Fires the Clean Air scent to clear potent smells.'''
    DURATION = 3.0
    if ATOMIZER_METHOD == "http":
        trigger_atomizer_http(6, DURATION)
    else:
        trigger_atomizer_raw_udp(6, DURATION)
    
def audio_callback(indata, frames, time, status):
    '''Callback function for processing audio data. This function is called whenever new audio data is available.'''
    if status:
        print(status)

    waveform = np.squeeze(indata).astype(np.float32)  # Remove single-dimensional entries from the shape of the array
    scores, embeddings, spectrogram = model(waveform)
    mean_scores = np.mean(scores, axis=0)
    top_class_index = np.argmax(mean_scores)
    inferred_class = labels[top_class_index]
    confidence = mean_scores[top_class_index]

    if confidence > 0.25:
        print(f"Detected sound: {inferred_class} (confidence: {confidence * 100:.1f}%)")

        if inferred_class in ["Gunshot, gunfire", "Machine gun", "Fusillade", ]:
            if ATOMIZER_METHOD == "http":
                trigger_atomizer_http(1, 1)
            else:
                trigger_atomizer_raw_udp(1, 1)
        elif inferred_class in ["Explosion", "Fireworks", "Firecracker", "Boom"]:
                    if ATOMIZER_METHOD == "http":
                        trigger_atomizer_http(2, 1)
                    else:
                        trigger_atomizer_raw_udp(2, 1)
        elif inferred_class in ["Motor vehicle (road)", "Car", "Skidding", "Tire squeal", "Race car, auto racing"]:
                    if ATOMIZER_METHOD == "http":
                        trigger_atomizer_http(3, 1)
                    else:
                        trigger_atomizer_raw_udp(3, 1)
        elif inferred_class in ["Rustling leaves", "Outside, rural or natural"]:
            if ATOMIZER_METHOD == "http":
                trigger_atomizer_http(4, 1)
            else:
                trigger_atomizer_raw_udp(4, 1)
        elif inferred_class in ["Thunderstorm", "Thunder", "Rain", "Raindrop", "Rain on surface"]:
                    if ATOMIZER_METHOD == "http":
                        trigger_atomizer_http(5, 1)
                    else:
                        trigger_atomizer_raw_udp(5, 1)


# Initialization
print("Starting GameStank...")
model = load_model()
labels = load_labels(model)

print("Starting live audio stream analysis. Press Ctrl-C to stop.")

with sd.InputStream(device=VB_CABLE, 
                    channels=1, 
                    samplerate=SAMPLE_RATE, 
                    blocksize=BLOCK_SIZE, 
                    callback=audio_callback(model=model, labels=labels)):
    while True:
         sd.sleep(1000)