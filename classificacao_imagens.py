import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# 🔹 Caminho para as imagens organizadas por classe
data_dir = "/content/drive/MyDrive/imagen_treino"

# 🔹 Parâmetros
batch_size = 32
img_height = 224
img_width = 224
seed = 123

# 🔹 Criar datasets de treinamento e validação
raw_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

raw_val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# 🔹 Pegar nomes das classes ANTES de mapear os datasets
class_names = raw_train_dataset.class_names
print("Classes detectadas:", class_names)

# 🔹 Normalização das imagens (0 a 1)
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_dataset = raw_train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = raw_val_dataset.map(lambda x, y: (normalization_layer(x), y))

# 🔹 Definição da CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')  # Saída com número de classes
])

# 🔹 Compilar o modelo
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 🔹 Treinar o modelo
epochs = 10
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)

# 🔹 Avaliação do modelo
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.figure(figsize=(8, 5))
plt.plot(range(epochs), acc, label="Treinamento")
plt.plot(range(epochs), val_acc, label="Validação")
plt.legend()
plt.title("Acurácia do Modelo")
plt.xlabel("Épocas")
plt.ylabel("Acurácia")
plt.show()

# 🔹 Função para testar uma imagem nova e exibir todas as previsões
def predict_image(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normaliza

    predictions = model.predict(img_array)[0]  # Pega a saída da rede neural
    sorted_indices = np.argsort(predictions)[::-1]  # Ordena previsões (maior para menor)

    # 🔹 Exibir a imagem
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Imagem de Teste")
    plt.show()

    # 🔹 Mostrar todas as previsões
    print("\n🔹 **Previsões para a imagem:**")
    for i in sorted_indices:
        print(f"{class_names[i]}: {predictions[i] * 100:.2f}%")

    # 🔹 Exibir gráfico de probabilidades
    plt.figure(figsize=(8, 5))
    plt.barh([class_names[i] for i in sorted_indices], [predictions[i] for i in sorted_indices], color="skyblue")
    plt.xlabel("Probabilidade")
    plt.title("Distribuição das previsões")
    plt.gca().invert_yaxis()
    plt.show()

# 🔹 Exemplo de teste com uma imagem nova
predict_image("/content/drive/MyDrive/imagem_teste/01-domesticated-dog.jpg", model)
