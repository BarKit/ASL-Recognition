import tensorflow as tf
import numpy as np
import os

def convert_model(old_model_path, new_model_path):
    """
    Konwertuje model do nowej wersji TensorFlow
    """
    try:
        print(f"Ładowanie modelu z {old_model_path}...")
        
        # Próba ładowania z różnymi opcjami
        model = None
        
        # Metoda 1: Standardowe ładowanie
        try:
            model = tf.keras.models.load_model(old_model_path)
            print("✓ Załadowano standardowo")
        except Exception as e:
            print(f"✗ Standardowe ładowanie nie powiodło się: {e}")
        
        # Metoda 2: Ładowanie bez kompilacji
        if model is None:
            try:
                model = tf.keras.models.load_model(old_model_path, compile=False)
                print("✓ Załadowano bez kompilacji")
            except Exception as e:
                print(f"✗ Ładowanie bez kompilacji nie powiodło się: {e}")
        
        # Metoda 3: Ładowanie tylko wag (wymaga rekonstrukcji architektury)
        if model is None:
            try:
                # Rekonstrukcja architektury (na podstawie Twojego kodu treningowego)
                model = create_model_architecture()
                model.load_weights(old_model_path)
                print("✓ Załadowano tylko wagi")
            except Exception as e:
                print(f"✗ Ładowanie wag nie powiodło się: {e}")
        
        if model is None:
            print("❌ Nie udało się załadować modelu")
            return False
        
        # Rekompilacja modelu
        print("Rekompilacja modelu...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Zapisanie w nowej wersji
        print(f"Zapisywanie modelu do {new_model_path}...")
        model.save(new_model_path)
        
        print("✅ Konwersja zakończona pomyślnie!")
        return True
        
    except Exception as e:
        print(f"❌ Błąd konwersji: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_model_architecture():
    """
    Odtwarza architekturę modelu na podstawie kodu treningowego
    """
    img_size = 64
    num_classes = 27
    
    model = tf.keras.Sequential()
    
    # Pierwsza warstwa konwolucyjna
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Druga warstwa konwolucyjna
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Trzecia warstwa konwolucyjna
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Czwarta warstwa konwolucyjna
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    
    # Spłaszczenie
    model.add(tf.keras.layers.Flatten())
    
    # Warstwy gęsto połączone
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    
    # Warstwa wyjściowa
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    return model

def test_model(model_path):
    """
    Testuje czy model można załadować i czy działa
    """
    try:
        print(f"Testowanie modelu {model_path}...")
        
        # Załadowanie modelu
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model załadowany")
        print(f"  - Kształt wejściowy: {model.input_shape}")
        print(f"  - Kształt wyjściowy: {model.output_shape}")
        
        # Test predykcji na losowych danych
        test_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
        predictions = model.predict(test_input, verbose=0)
        
        print(f"✓ Predykcja działa")
        print(f"  - Kształt wyjścia: {predictions.shape}")
        print(f"  - Suma prawdopodobieństw: {np.sum(predictions):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test nie powiódł się: {e}")
        return False

if __name__ == "__main__":
    # Ścieżki
    old_model_path = "ASL-Recognition/WEBAPP/asl_model.h5"
    new_model_path = "ASL-Recognition/WEBAPP/asl_model_converted.h5"
    
    print("🔧 Konwerter modelu ASL")
    print("=" * 50)
    
    # Sprawdź czy stary model istnieje
    if not os.path.exists(old_model_path):
        print(f"❌ Plik {old_model_path} nie istnieje!")
        exit(1)
    
    # Informacje o wersji TensorFlow
    print(f"Wersja TensorFlow: {tf.__version__}")
    
    # Konwertuj model
    if convert_model(old_model_path, new_model_path):
        print("\n" + "=" * 50)
        print("🧪 Testowanie skonwertowanego modelu...")
        test_model(new_model_path)
        
        print("\n" + "=" * 50)
        print("💡 Instrukcje:")
        print(f"1. Zmień MODEL_PATH w app.py na: '{new_model_path}'")
        print("2. Uruchom ponownie aplikację")
    else:
        print("\n❌ Konwersja nie powiodła się")