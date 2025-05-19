import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Konfiguracja GPU
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"Znaleziono {len(physical_devices)} urządzeń GPU:")
    for device in physical_devices:
        print(f" - {device}")
    # Konfiguracja GPU - pozwól TensorFlow na alokację pamięci dynamicznie
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Ustawiono dynamiczną alokację pamięci GPU")
    except RuntimeError as e:
        print(f"Błąd konfiguracji GPU: {e}")
else:
    print("UWAGA: Nie znaleziono urządzeń GPU. Model będzie trenowany na CPU, co może być znacznie wolniejsze.")

# Stałe
img_size = 64  # rozmiar obrazu po przeskalowaniu
num_classes = 27  # 26 liter alfabetu + space

def load_dataset_from_directory(train_dir, test_dir):
    """
    Ładuje zbiór danych ASL z podanej struktury katalogów, gdzie
    train_dir i test_dir zawierają podfoldery dla każdej klasy.
    """
    # Mapowanie etykiet
    label_map = {}
    for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
        label_map[letter] = i
    
    # Dodajemy specjalną wartość dla spacji (znak nr 26)
    label_map['space'] = 26
    
    # Funkcja pomocnicza do wczytywania obrazów z danego katalogu
    def load_images_from_dir(directory):
        images = []
        labels = []
        
        # Dla każdego podfolderu (klasy) w katalogu
        for folder in os.listdir(directory):
            folder_path = os.path.join(directory, folder)
            if os.path.isdir(folder_path):
                label = label_map.get(folder)
                if label is not None:  # tylko jeśli folder jest w mapowaniu
                    print(f"Wczytywanie danych z katalogu: {folder_path}")
                    for img_file in os.listdir(folder_path):
                        if img_file.endswith('.jpg') or img_file.endswith('.png'):
                            img_path = os.path.join(folder_path, img_file)
                            try:
                                # Wczytanie i przeskalowanie obrazu
                                img = cv2.imread(img_path)
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konwersja z BGR na RGB
                                img = cv2.resize(img, (img_size, img_size))
                                
                                # Normalizacja wartości pikseli do przedziału [0, 1]
                                img = img / 255.0
                                
                                # Dodanie do listy
                                images.append(img)
                                labels.append(label)
                            except Exception as e:
                                print(f"Błąd podczas przetwarzania {img_path}: {e}")
        
        # Konwersja list na tablice numpy
        X = np.array(images)
        y = np.array(labels)
        
        # Konwersja etykiet na format one-hot encoding
        y = to_categorical(y, num_classes)
        
        return X, y
    
    # Wczytanie danych treningowych i testowych
    print("Wczytywanie danych treningowych...")
    X_train, y_train = load_images_from_dir(train_dir)
    print(f"Załadowano {X_train.shape[0]} obrazów treningowych")
    
    print("Wczytywanie danych testowych...")
    X_test, y_test = load_images_from_dir(test_dir)
    print(f"Załadowano {X_test.shape[0]} obrazów testowych")
    
    return X_train, y_train, X_test, y_test

def create_model():
    """
    Tworzy model CNN do klasyfikacji ASL.
    """
    model = Sequential()
    
    # Pierwsza warstwa konwolucyjna
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Druga warstwa konwolucyjna
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Trzecia warstwa konwolucyjna
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Czwarta warstwa konwolucyjna
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    # Spłaszczenie
    model.add(Flatten())
    
    # Warstwy gęsto połączone
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    # Warstwa wyjściowa
    model.add(Dense(num_classes, activation='softmax'))
    
    # Kompilacja modelu
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data_generators(X_train, y_train, X_val=None, y_val=None):
    """
    Przygotowuje generatory danych z augmentacją dla zbioru treningowego.
    Jeśli nie podano zbioru walidacyjnego, zostanie on wygenerowany z treningowego.
    """
    # Generator z augmentacją dla zbioru treningowego
    train_datagen = ImageDataGenerator(
        rotation_range=10,      # losowe obroty o kąt ±10 stopni
        width_shift_range=0.1,  # losowe przesunięcia w poziomie o 10%
        height_shift_range=0.1, # losowe przesunięcia w pionie o 10%
        zoom_range=0.1,         # losowe przybliżenia o 10%
        brightness_range=[0.9, 1.1],  # losowe zmiany jasności o ±10%
        horizontal_flip=False,  # bez odbić w poziomie (zmieniłyby znaczenie niektórych znaków)
        fill_mode='nearest',    # metoda wypełniania nowych pikseli
        validation_split=0.15 if X_val is None else None  # podział na zbiór walidacyjny, jeśli nie podano X_val
    )
    
    if X_val is None:
        # Tworzenie generatorów treningowego i walidacyjnego z podziałem wewnętrznym
        train_generator = train_datagen.flow(
            X_train, y_train, 
            batch_size=32,
            subset='training'
        )
        
        val_generator = train_datagen.flow(
            X_train, y_train, 
            batch_size=32,
            subset='validation'
        )
    else:
        # Użycie podanego zbioru walidacyjnego
        train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
        
        # Generator dla zbioru walidacyjnego (bez augmentacji)
        val_datagen = ImageDataGenerator()
        val_generator = val_datagen.flow(X_val, y_val, batch_size=32)
    
    return train_generator, val_generator

def train_model(model, train_generator, val_generator, epochs=30):
    """
    Trenuje model CNN.
    """
    # Definiowanie callbacków
    checkpoint = ModelCheckpoint(
        'best_asl_model.h5',
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=0.00001
    )
    
    callbacks = [checkpoint, early_stopping, reduce_lr]
    
    # Trenowanie modelu
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, X_test, y_test):
    """
    Ocenia model na zbiorze testowym.
    """
    # Ocena modelu
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f'Test accuracy: {test_accuracy:.4f}')
    
    # Predykcje
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Raport klasyfikacji
    class_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space']
    report = classification_report(y_true_classes, y_pred_classes, target_names=class_names)
    print(report)
    
    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(20, 20))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Macierz pomyłek')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Dodanie wartości liczbowych do macierzy
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, conf_matrix[i, j],
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('Rzeczywista klasa')
    plt.xlabel('Predykowana klasa')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    return test_accuracy, report, conf_matrix

def plot_training_history(history):
    """
    Wizualizuje historię treningu modelu.
    """
    plt.figure(figsize=(12, 4))
    
    # Wykres dokładności
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_predictions(model, X_test, y_test, num_samples=10):
    """
    Wizualizuje przykładowe predykcje modelu.
    """
    # Losowy wybór próbek
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    X_samples = X_test[indices]
    y_true = np.argmax(y_test[indices], axis=1)
    
    # Predykcje
    y_pred = model.predict(X_samples)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Mapowanie indeksów na litery
    class_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['space']
    
    # Wizualizacja
    plt.figure(figsize=(20, 10))
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_samples[i])
        plt.title(f'True: {class_names[y_true[i]]}\nPred: {class_names[y_pred_classes[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()

def main():
    # Ścieżka do zbioru danych
    dataset_path = 'ASL_Dataset'  # Katalog zawierający foldery Train i Test
    train_dir = os.path.join(dataset_path, 'Train')
    test_dir = os.path.join(dataset_path, 'Test')
    
    print("Ładowanie danych...")
    X_train, y_train, X_test, y_test = load_dataset_from_directory(train_dir, test_dir)
    print(f"Załadowano dane treningowe: {X_train.shape} i testowe: {X_test.shape}")
    
    # Przygotowanie generatorów danych (bez jawnego zbioru walidacyjnego - zostanie wygenerowany wewnętrznie)
    train_generator, val_generator = prepare_data_generators(X_train, y_train)
    
    # Tworzenie modelu
    print("Tworzenie modelu CNN...")
    model = create_model()
    model.summary()
    
    # Trenowanie modelu
    print("Rozpoczęcie treningu...")
    history = train_model(model, train_generator, val_generator)
    
    # Wizualizacja historii treningu
    plot_training_history(history)
    
    # Ocena modelu
    print("Ocena modelu na zbiorze testowym...")
    test_accuracy, report, conf_matrix = evaluate_model(model, X_test, y_test)
    
    # Wizualizacja przykładowych predykcji
    visualize_predictions(model, X_test, y_test)
    
    # Zapisanie modelu
    model.save('asl_recognition_model.h5')
    print("Model został zapisany jako 'asl_recognition_model.h5'")
    
    return model  # Zwracamy model, aby był dostępny po uruchomieniu funkcji

if __name__ == "__main__":
    main()