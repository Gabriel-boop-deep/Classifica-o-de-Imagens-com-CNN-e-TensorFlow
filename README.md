# 📌 Classificação de Imagens com CNN e TensorFlow

Este repositório apresenta um modelo de rede neural convolucional (CNN) para classificação de imagens, utilizando TensorFlow e Keras. O modelo recebe imagens organizadas por classes, treina uma CNN e permite a previsão em novas imagens.

---

## 📂 Estrutura do Projeto

```
/
├── dataset/  # Diretório contendo imagens organizadas por classe
│   ├── classe_1/
│   ├── classe_2/
│   ├── ...
│
├── imagem_teste/  # Imagens para teste do modelo
│   ├── teste1.jpg
│   ├── teste2.jpg
│
├── modelo/  # Diretório para armazenar pesos do modelo treinado
│
├── train.py  # Script para treinamento do modelo
├── predict.py  # Script para previsão de imagens
└── README.md  # Documentação do projeto
```

---

## 🔧 Configuração do Ambiente

Antes de rodar o código, instale as dependências necessárias:

```bash
pip install tensorflow numpy matplotlib
```

Se for usar o Google Drive para armazenar os dados, monte o drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 📥 Carregamento e Processamento dos Dados

O dataset é carregado automaticamente do diretório indicado, com as imagens redimensionadas para `224x224` e divididas em conjuntos de treinamento e validação:

```python
raw_train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
```

O mesmo processo é feito para a validação, garantindo que 20% das imagens sejam separadas para testes:

```python
raw_val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)
```

As imagens são normalizadas para o intervalo `[0,1]` antes do treinamento:

```python
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = raw_train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = raw_val_dataset.map(lambda x, y: (normalization_layer(x), y))
```

---

## 🏗 Definição da CNN

A rede neural convolucional é composta por:
- 3 camadas convolucionais com ReLU
- 3 camadas de pooling
- 1 camada totalmente conectada (dense)
- Camada de saída com `softmax` para classificação

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])
```

O modelo é compilado com o otimizador `adam` e a função de perda `sparse_categorical_crossentropy`:

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## 🚀 Treinamento do Modelo

O modelo é treinado por 10 épocas, utilizando os datasets de treinamento e validação:

```python
epochs = 10
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs
)
```

---

## 📊 Avaliação e Visualização da Performance

A acurácia do modelo é plotada ao longo das épocas para monitoramento do desempenho:

```python
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
```

---

## 🔍 Teste com Imagem Nova

Para realizar previsões em uma imagem nova:

```python
def predict_image(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)[0]
    sorted_indices = np.argsort(predictions)[::-1]
    
    plt.figure(figsize=(6, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Imagem de Teste")
    plt.show()
    
    print("\n🔹 **Previsões para a imagem:**")
    for i in sorted_indices:
        print(f"{class_names[i]}: {predictions[i] * 100:.2f}%")
    
    plt.figure(figsize=(8, 5))
    plt.barh([class_names[i] for i in sorted_indices], [predictions[i] for i in sorted_indices], color="skyblue")
    plt.xlabel("Probabilidade")
    plt.title("Distribuição das previsões")
    plt.gca().invert_yaxis()
    plt.show()
```

Para testar:

```python
predict_image("/content/drive/MyDrive/imagem_teste/01-domesticated-dog.jpg", model)
```

---

## 📌 Conclusão

Este projeto demonstra um pipeline completo para classificação de imagens usando redes neurais convolucionais com TensorFlow. Ele pode ser expandido para incluir aumento de dados, fine-tuning com redes pré-treinadas e otimização de hiperparâmetros.

---

🔗 **Autor:** Gabriel Nunes Barbosa Nogueira  
📧 **Contato:** gabrielnbn@hotmail.com

