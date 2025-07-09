from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import cv2
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# Konfiguracja
MODEL_PATH = 'WEBAPP/asl_model.h5'
IMAGE_SIZE = (64, 64)

# Mapowanie klas (dostosuj do swojego modelu)
CLASS_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SPACE'
]

# Załaduj model przy starcie aplikacji
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model załadowany pomyślnie z {MODEL_PATH}")
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {e}")
    model = None

def preprocess_image(image_data):
    """Przetwarzanie obrazu do formatu wymaganego przez model"""
    try:
        # Dekodowanie base64
        image_data = image_data.split(',')[1]  # Usuń prefix "data:image/jpeg;base64,"
        image_bytes = base64.b64decode(image_data)
        
        # Konwersja do OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Nie można zdekodować obrazu")
        
        # Konwersja BGR do RGB (OpenCV używa BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Wykryj region dłoni (Region of Interest - ROI)
        # Możesz dostosować te współrzędne do lepszego kadrowania dłoni
        height, width = image.shape[:2]
        
        # Wytnij centralny kwadrat (lepiej dla gestów dłoni)
        center_x, center_y = width // 2, height // 2
        roi_size = min(width, height) // 2
        
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(width, center_x + roi_size // 2)
        y2 = min(height, center_y + roi_size // 2)
        
        roi = image[y1:y2, x1:x2]
        
        # Zmiana rozmiaru do wymaganego przez model
        image_resized = cv2.resize(roi, IMAGE_SIZE)
        
        # Konwersja do numpy array
        image_array = np.array(image_resized, dtype=np.float32)
        
        # Normalizacja - sprawdź czy Twój model był trenowany z normalizacją 0-1 czy -1,1
        # Dla większości modeli CNN: 0-1
        image_array = image_array / 255.0
        
        # Niektóre modele mogą wymagać normalizacji -1,1:
        # image_array = (image_array / 127.5) - 1.0
        
        # Dodaj wymiar batch
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        print(f"Błąd przetwarzania obrazu: {e}")
        return None

@app.route('/')
def index():
    """Strona główna"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint do predykcji"""
    if model is None:
        return jsonify({'error': 'Model nie został załadowany'}), 500
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Brak danych obrazu'}), 400
        
        # Przetwórz obraz
        processed_image = preprocess_image(image_data)
        if processed_image is None:
            return jsonify({'error': 'Błąd przetwarzania obrazu'}), 400
        
        # Wykonaj predykcję
        predictions = model.predict(processed_image, verbose=0)
        
        # Pobierz najlepszą predykcję
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_letter = CLASS_NAMES[predicted_class_idx]
        
        # Ustaw minimalny próg pewności
        MIN_CONFIDENCE = 0.6
        if confidence < MIN_CONFIDENCE:
            predicted_letter = "?"
            confidence = 0.0
        
        # Pobierz top 3 predykcje
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = [
            {
                'letter': CLASS_NAMES[i],
                'confidence': float(predictions[0][i])
            }
            for i in top_indices
        ]
        
        # Debug: wypisz wszystkie predykcje
        print(f"Predykcje: {[(CLASS_NAMES[i], float(predictions[0][i])) for i in range(len(CLASS_NAMES))]}")
        print(f"Najlepsza: {predicted_letter} ({confidence:.3f})")
        
        return jsonify({
            'predicted_letter': predicted_letter,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'all_predictions': [float(p) for p in predictions[0]]  # Do debugowania
        })
    
    except Exception as e:
        print(f"Błąd predykcji: {e}")
        return jsonify({'error': f'Błąd predykcji: {str(e)}'}), 500

@app.route('/health')
def health():
    """Sprawdzenie stanu aplikacji"""
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH
    })

if __name__ == '__main__':
    # Upewnij się, że folder templates istnieje
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)