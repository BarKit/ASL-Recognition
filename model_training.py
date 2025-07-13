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

# Stałe
img_size = 64  # rozmiar obrazu po przeskalowaniu
num_classes = 27  # 26 liter alfabetu + space

# Mapowanie etykiet - definicja globalna dla spójności
LABEL_MAP = {}
for i, letter in enumerate('ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    LABEL_MAP[letter] = i
# Dodajemy specjalną wartość dla spacji (znak nr 26)
LABEL_MAP['Space'] = 26

# Lista nazw klas w odpowiedniej kolejności
CLASS_NAMES = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + ['Space']

def prepare_data():
    """
    Przygotowuje dane treningowe, walidacyjne i testowe korzystając z ImageDataGenerator.

    """
    # Ścieżki do zbiorów danych
    dataset_path = 'ASL_Dataset'
    train_dir = os.path.join(dataset_path, 'Train')
    test_dir = os.path.join(dataset_path, 'Test')
    
    print("Przygotowanie generatorów danych...")
    
    # Generator z augmentacją dla zbioru treningowego (90% z Train)
    train_datagen = ImageDataGenerator(
        rescale=1./255,             # normalizacja wartości pikseli
        rotation_range=10,          # losowe obroty o kąt ±10 stopni
        width_shift_range=0.1,      # losowe przesunięcia w poziomie o 10%
        height_shift_range=0.1,     # losowe przesunięcia w pionie o 10%
        zoom_range=0.1,             # losowe przybliżenia o 10%
        brightness_range=[0.9, 1.1], # losowe zmiany jasności o ±10%
        horizontal_flip=True,       # dodane odbicie lustrzane, aby model rozpoznawał gesty prawej ręki
        validation_split=0.1,       # 10% danych treningowych używamy jako walidacyjne
        fill_mode='nearest'         # metoda wypełniania nowych pikseli
    )
    
    # Generator dla zbioru testowego (tylko skalowanie)
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    
    # Tworzenie generatora treningowego
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Tworzenie generatora walidacyjnego
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    # Tworzenie generatora testowego
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical',
        shuffle=False  # Bez mieszania dla testu, aby zachować właściwą kolejność dla ewaluacji
    )
    
    print(f"Przygotowano {train_generator.samples} próbek treningowych")
    print(f"Przygotowano {validation_generator.samples} próbek walidacyjnych")
    print(f"Przygotowano {test_generator.samples} próbek testowych")
    
    # Wyświetlenie przykładowych obrazów po augmentacji
    def show_augmented_images(generator, num_images=5):
        """Wyświetla przykładowe augmentowane obrazy z generatora."""
        plt.figure(figsize=(15, 3))
        
        # Pobierz jedną partię danych
        images, labels = next(generator)
        
        for i in range(min(num_images, len(images))):
            plt.subplot(1, num_images, i+1)
            # Konwersja z [0,1] do [0,255] dla wizualizacji
            img = images[i] * 255
            img = img.astype(np.uint8)
            plt.imshow(img)
            label_idx = np.argmax(labels[i])
            class_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else f"Unknown({label_idx})"
            plt.title(f"Class: {class_name}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('augmented_samples.png')
        plt.show()
    
    # Wyświetl przykładowe obrazy po augmentacji
    print("Przykładowe obrazy po augmentacji:")
    show_augmented_images(train_generator)
    
    return train_generator, validation_generator, test_generator

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

def train_model(model, train_generator, validation_generator, epochs=15):
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
        patience=3,
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
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    
    return history

def evaluate_model(model, test_generator):
    """
    Ocenia model na zbiorze testowym.
    """
    # Ocena modelu
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test accuracy: {test_accuracy:.4f}')
    
    # Predykcje
    y_pred_prob = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred_prob, axis=1)
    
    # Prawdziwe etykiety
    # Uwzględnienie, że generator zwraca dane w partiach
    y_true_classes = test_generator.classes
    
    # Sprawdzenie, które klasy są obecne w danych testowych
    present_class_indices = sorted(np.unique(y_true_classes))
    print(f"Liczba unikalnych klas w zbiorze testowym: {len(present_class_indices)}")
    print(f"Indeksy obecnych klas: {present_class_indices}")
    
    # Tworzenie nazw klas dla obecnych indeksów
    present_class_names = [CLASS_NAMES[i] for i in present_class_indices]
    print(f"Obecne klasy: {present_class_names}")
    
    # Raport klasyfikacji
    report = classification_report(
        y_true_classes, 
        y_pred_classes, 
        labels=present_class_indices,
        target_names=present_class_names
    )
    print(report)
    
    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_true_classes, y_pred_classes, labels=present_class_indices)
    
    # Wizualizacja macierzy pomyłek
    plt.figure(figsize=(20, 20))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Macierz pomyłek')
    plt.colorbar()
    tick_marks = np.arange(len(present_class_names))
    plt.xticks(tick_marks, present_class_names, rotation=45)
    plt.yticks(tick_marks, present_class_names)
    
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
    plt.title('Dokładność w trakcie treningu i walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    
    # Wykres straty
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Strata w trakcie treningu i walidacji')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_predictions(model, test_generator, num_samples=10):
    """
    Wizualizuje przykładowe predykcje modelu.
    """
    # Pobierz próbki z generatora testowego
    test_generator.reset()  # Zresetowanie generatora do początku
    
    # Pobierz jedną partię danych
    images, labels = next(test_generator)
    
    # Predykcje
    y_pred = model.predict(images)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(labels, axis=1)
    
    # Wizualizacja
    plt.figure(figsize=(20, 10))
    for i in range(min(num_samples, len(images))):
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i])
        true_label = CLASS_NAMES[y_true_classes[i]] if y_true_classes[i] < len(CLASS_NAMES) else f"Unknown({y_true_classes[i]})"
        pred_label = CLASS_NAMES[y_pred_classes[i]] if y_pred_classes[i] < len(CLASS_NAMES) else f"Unknown({y_pred_classes[i]})"
        plt.title(f'True: {true_label}\nPred: {pred_label}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.show()

def main():
    """
    Główna funkcja programu.
    """
    print("Przygotowanie danych...")
    train_generator, validation_generator, test_generator = prepare_data()
    
    # Tworzenie modelu
    print("Tworzenie modelu CNN...")
    model = create_model()
    model.summary()
    
    # Trenowanie modelu
    print("Rozpoczęcie treningu...")
    history = train_model(model, train_generator, validation_generator)
    
    # Wizualizacja historii treningu
    plot_training_history(history)
    
    # Ocena modelu
    print("Ocena modelu na zbiorze testowym...")
    test_accuracy, report, conf_matrix = evaluate_model(model, test_generator)
    
    # Wizualizacja przykładowych predykcji
    visualize_predictions(model, test_generator)
    
    # Zapisanie modelu
    model.save('asl_recognition_model.h5')
    print("Model został zapisany jako 'asl_recognition_model.h5'")
    
    return model  # Zwracamy model, aby był dostępny po uruchomieniu funkcji

if __name__ == "__main__":
    main()