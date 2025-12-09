import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- KONFIGURACJA ---
IMG_SIZE = 128
BATCH_SIZE = 32
DATASET_PATH = 'ASL_Dataset'
OUTPUT_DIR = 'Training_Logs'
MODEL_FILENAME = 'asl_model_128.h5'

def prepare_data():
    """Przygotowuje dane i tworzy folder na logi."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Utworzono folder na logi: {OUTPUT_DIR}")

    train_dir = os.path.join(DATASET_PATH, 'Train')
    test_dir = os.path.join(DATASET_PATH, 'Test')
    
    print(f"Przygotowanie generatorów danych (Rozdzielczość: {IMG_SIZE}x{IMG_SIZE})...")
    
    # Augmentacja danych dla treningu
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        validation_split=0.1,
        fill_mode='nearest'
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Mapowanie klas
    class_indices = train_generator.class_indices
    label_map = {v: k for k, v in class_indices.items()}
    num_classes = len(label_map)
    
    # Zapis mapowania do pliku dla pewności
    with open(os.path.join(OUTPUT_DIR, 'class_map.txt'), 'w') as f:
        f.write(str(label_map))
        
    return train_generator, validation_generator, test_generator, label_map, num_classes

def create_model(num_classes):
    """Tworzy model CNN."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        Conv2D(256, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        Conv2D(512, kernel_size=(3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator, epochs=15):
    """Trenuje model."""
    checkpoint_path = 'best_' + MODEL_FILENAME
    
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_loss', patience=3, verbose=1, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1, min_lr=0.00001)
    ]
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )
    return history

def plot_training_history(history):
    """Generuje wykresy treningu."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Dokładność')
    plt.xlabel('Epoka')
    plt.ylabel('Dokładność')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Strata')
    plt.xlabel('Epoka')
    plt.ylabel('Strata')
    plt.legend()
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'training_history.png')
    plt.savefig(save_path)
    print(f"Zapisano historię treningu do: {save_path}")
    plt.close()

def evaluate_model(model, test_generator, label_map):
    """Ocenia model i generuje macierz pomyłek."""
    print("\n--- Ewaluacja Modelu ---")
    loss, accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {accuracy:.4f}')
    
    # Reset generatora i predykcje
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    class_names = [label_map[i] for i in range(len(label_map))]
    
    # Raport
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    print(report)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Macierz pomyłek
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(20, 20))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Macierz pomyłek')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"Zapisano macierz pomyłek do: {save_path}")
    plt.close()

def visualize_predictions(model, label_map):
    """
    Generuje prediction_samples.png pobierając losowe pliki z całego zbioru testowego.
    """
    print("\n--- Generowanie wizualizacji predykcji ---")
    test_dir = os.path.join(DATASET_PATH, 'Test')
    
    # 1. Zbieramy wszystkie możliwe ścieżki do plików w zbiorze testowym
    all_test_files = []
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    all_test_files.append({
                        'path': os.path.join(class_dir, filename),
                        'true_label': class_name
                    })
    
    # 2. Losujemy 10 unikalnych zdjęć z całej puli
    if len(all_test_files) < 10:
        print("Za mało plików w zbiorze testowym do wizualizacji.")
        return

    selected_samples = random.sample(all_test_files, 10)
    
    plt.figure(figsize=(20, 10))
    
    for i, sample in enumerate(selected_samples):
        # Ładowanie i preprocessing obrazu
        img = load_img(sample['path'], target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_processed = img_array / 255.0  # Normalizacja
        img_batch = np.expand_dims(img_processed, axis=0) # Dodanie wymiaru batch
        
        # Predykcja
        prediction = model.predict(img_batch, verbose=0)
        pred_idx = np.argmax(prediction)
        
        # Mapowanie indeksu na nazwę (zabezpieczenie przed brakiem klucza)
        pred_label = label_map.get(pred_idx, "Unknown")
        true_label = sample['true_label']
        
        # Kolor tekstu (Zielony=OK, Czerwony=Błąd)
        color = 'green' if true_label == pred_label else 'red'
        
        # Rysowanie
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color, fontsize=14, fontweight='bold')
        plt.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, 'prediction_samples.png')
    plt.savefig(save_path)
    print(f"Zapisano próbki predykcji do: {save_path}")
    plt.close()

def generate_augmented_samples_preview(train_generator):
    """Zapisuje podgląd augmentacji."""
    images, _ = next(train_generator)
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(images))):
        plt.subplot(1, 5, i+1)
        plt.imshow(images[i])
        plt.axis('off')
        plt.title("Augmented Sample")
    save_path = os.path.join(OUTPUT_DIR, 'augmented_samples.png')
    plt.savefig(save_path)
    plt.close()

def main():
    # Przygotowanie danych
    train_gen, val_gen, test_gen, label_map, num_classes = prepare_data()
    
    # Generowanie podglądu augmentacji
    generate_augmented_samples_preview(train_gen)
    
    # Model
    model = create_model(num_classes)
    
    print("\nRozpoczęcie treningu...")
    history = train_model(model, train_gen, val_gen)
    
    plot_training_history(history)
    
    print(f"\nŁadowanie najlepszego modelu (best_{MODEL_FILENAME})...")
    model.load_weights('best_' + MODEL_FILENAME)
    
    # Ewaluacja i wizualizacja
    evaluate_model(model, test_gen, label_map)
    visualize_predictions(model, label_map)
    
    # Zapis modelu końcowego
    model.save(MODEL_FILENAME)
    print(f"Zapisano model końcowy jako '{MODEL_FILENAME}' w głównym katalogu.")

if __name__ == "__main__":
    main()