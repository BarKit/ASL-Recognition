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
MODEL_PATH = 'ASL-Recognition/WEBAPP/asl_model_converted.h5'
IMAGE_SIZE = (64, 64)

# Mapowanie klas (dostosuj do swojego modelu)
CLASS_NAMES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['Space']

# Załaduj model przy starcie aplikacji
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model załadowany pomyślnie z {MODEL_PATH}")
    print(f"Architektura modelu: {model.input_shape} -> {model.output_shape}")
except Exception as e:
    print(f"Błąd podczas ładowania modelu: {e}")
    model = None

def preprocess_image(image_data):
    """
    UPROSZCZONE przetwarzanie obrazu - zgodne z treningiem
    """
    try:
        # Dekodowanie base64
        image_data = image_data.split(',')[1]  # Usuń prefix "data:image/jpeg;base64,"
        image_bytes = base64.b64decode(image_data)
        
        # Konwersja do PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Konwersja do RGB jeśli potrzebne
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Konwersja do numpy array
        image_array = np.array(image)
        
        # UPROSZCZONE przetwarzanie - tylko resize jak w treningu
        image_resized = cv2.resize(image_array, IMAGE_SIZE)
        
        # Normalizacja IDENTYCZNA jak w treningu
        image_array = image_resized.astype(np.float32) / 255.0
        
        # Dodaj wymiar batch
        image_array = np.expand_dims(image_array, axis=0)
        
        # Debug - sprawdź kształt i zakres wartości
        print(f"Kształt obrazu: {image_array.shape}")
        print(f"Zakres wartości: {image_array.min():.3f} - {image_array.max():.3f}")
        
        return image_array
    
    except Exception as e:
        print(f"Błąd przetwarzania obrazu: {e}")
        return None

def preprocess_image_with_roi(image_data):
    """
    Alternatywna wersja z ROI - do testowania
    """
    try:
        # Dekodowanie base64
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Konwersja do OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Nie można zdekodować obrazu")
        
        # Konwersja BGR do RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # ROI - centralny kwadrat
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        roi_size = min(width, height) // 2
        
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(width, center_x + roi_size // 2)
        y2 = min(height, center_y + roi_size // 2)
        
        roi = image[y1:y2, x1:x2]
        
        # Resize i normalizacja
        image_resized = cv2.resize(roi, IMAGE_SIZE)
        image_array = image_resized.astype(np.float32) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        print(f"Błąd przetwarzania obrazu z ROI: {e}")
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
        use_roi = data.get('use_roi', False)  # Opcjonalny parametr
        
        if not image_data:
            return jsonify({'error': 'Brak danych obrazu'}), 400
        
        # Przetwórz obraz - wypróbuj obie metody
        if use_roi:
            processed_image = preprocess_image_with_roi(image_data)
        else:
            processed_image = preprocess_image(image_data)
            
        if processed_image is None:
            return jsonify({'error': 'Błąd przetwarzania obrazu'}), 400
        
        # Wykonaj predykcję
        predictions = model.predict(processed_image, verbose=0)
        
        # Debug - wypisz surowe predykcje
        print(f"Surowe predykcje: {predictions[0]}")
        
        # Pobierz najlepszą predykcję
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        # Sprawdź czy indeks jest w zakresie
        if predicted_class_idx >= len(CLASS_NAMES):
            predicted_letter = f"UNKNOWN_CLASS_{predicted_class_idx}"
        else:
            predicted_letter = CLASS_NAMES[predicted_class_idx]
        
        # ZMNIEJSZONY próg pewności dla testów
        MIN_CONFIDENCE = 0.3
        if confidence < MIN_CONFIDENCE:
            predicted_letter = "?"
            confidence = 0.0
        
        # Pobierz top 3 predykcje
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        
        for i in top_indices:
            if i < len(CLASS_NAMES):
                top_predictions.append({
                    'letter': CLASS_NAMES[i],
                    'confidence': float(predictions[0][i])
                })
        
        # Szczegółowe logowanie
        print(f"Predykcja: {predicted_letter} (indeks: {predicted_class_idx})")
        print(f"Pewność: {confidence:.3f}")
        print(f"Top 3: {[(p['letter'], p['confidence']) for p in top_predictions]}")
        
        return jsonify({
            'predicted_letter': predicted_letter,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'predicted_class_idx': int(predicted_class_idx),
            'total_classes': len(CLASS_NAMES),
            'processing_method': 'ROI' if use_roi else 'Simple'
        })
    
    except Exception as e:
        print(f"Błąd predykcji: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Błąd predykcji: {str(e)}'}), 500

@app.route('/health')
def health():
    """Sprawdzenie stanu aplikacji"""
    model_info = {}
    if model is not None:
        model_info = {
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': model.count_params()
        }
    
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'model_path': MODEL_PATH,
        'model_info': model_info,
        'class_names': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES)
    })

@app.route('/test_processing', methods=['POST'])
def test_processing():
    """Endpoint do testowania obu metod przetwarzania"""
    if model is None:
        return jsonify({'error': 'Model nie został załadowany'}), 500
    
    try:
        data = request.json
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'error': 'Brak danych obrazu'}), 400
        
        # Testuj obie metody
        simple_processed = preprocess_image(image_data)
        roi_processed = preprocess_image_with_roi(image_data)
        
        results = {}
        
        if simple_processed is not None:
            simple_pred = model.predict(simple_processed, verbose=0)
            simple_idx = np.argmax(simple_pred[0])
            results['simple'] = {
                'letter': CLASS_NAMES[simple_idx] if simple_idx < len(CLASS_NAMES) else 'UNKNOWN',
                'confidence': float(simple_pred[0][simple_idx]),
                'top_3': [float(x) for x in np.sort(simple_pred[0])[-3:][::-1]]
            }
        
        if roi_processed is not None:
            roi_pred = model.predict(roi_processed, verbose=0)
            roi_idx = np.argmax(roi_pred[0])
            results['roi'] = {
                'letter': CLASS_NAMES[roi_idx] if roi_idx < len(CLASS_NAMES) else 'UNKNOWN',
                'confidence': float(roi_pred[0][roi_idx]),
                'top_3': [float(x) for x in np.sort(roi_pred[0])[-3:][::-1]]
            }
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Błąd testowania: {e}")
        return jsonify({'error': f'Błąd testowania: {str(e)}'}), 500

if __name__ == '__main__':
    # Sprawdź czy model istnieje
    if not os.path.exists(MODEL_PATH):
        print(f"UWAGA: Plik modelu {MODEL_PATH} nie istnieje!")
        print("Dostępne pliki .h5:")
        for file in os.listdir('.'):
            if file.endswith('.h5'):
                print(f"  - {file}")
    
    # Upewnij się, że folder templates istnieje
    os.makedirs('templates', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)