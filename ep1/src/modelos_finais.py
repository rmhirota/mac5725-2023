import pandas as pd
import numpy as np
from tensorflow import string
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import (
    TextVectorization,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LambdaCallback


def ler_dados(path_dados):
    dados = pd.read_csv(path_dados)["texto"]
    return dados


def ler_labels(path_label):
    dados = pd.read_csv(path_label)["rotulo"]
    labels = to_categorical(dados, num_classes=6)
    return labels


def tokens(tammax, dados):
    layer_tokens = TextVectorization(
        output_mode="int",
        output_sequence_length=tammax,
        pad_to_max_tokens=True,
        max_tokens=20000,
    )
    layer_tokens.adapt(dados)
    return layer_tokens


def print_metrics(epoch, logs, bidirecional):
    loss = logs["loss"]
    accuracy = logs["accuracy"]
    val_loss = logs["val_loss"]
    val_accuracy = logs["val_accuracy"]

    # Print metrics
    print(
        f"Epoch {epoch + 1} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f}"
    )

    # Save metrics to a file
    with open(f"metrics{bidirecional}.csv", "a") as file:
        file.write(
            f"Epoch {epoch + 1}, Loss {loss:.4f}, Accuracy {accuracy:.4f}, Val Loss {val_loss:.4f}, Val Accuracy {val_accuracy:.4f}\n"
        )


save_metrics_callback = LambdaCallback(
    on_epoch_end=lambda epoch, logs: print_metrics(epoch, logs, bidirecional)
    if (epoch + 1) % 5 == 0
    else None
)

dados_treino = ler_dados("base_treino.csv")
dados_validacao = ler_dados("base_validacao.csv")
dados_teste = ler_dados("base_teste.csv")

labels_treino = ler_labels("base_treino.csv")
labels_validacao = ler_labels("base_validacao.csv")
labels_teste = ler_labels("base_teste.csv")

tammax = 100
batch_size = 256

# Modelo com encoder unidirecional ---


def gerar_modelo(dropout, layer_tokens, bidirecional):
    if bidirecional == 0:
        model = Sequential(
            [
                Input(shape=(1,), dtype=string),
                layer_tokens,
                Embedding(input_dim=layer_tokens.vocabulary_size(), output_dim=32),
                LSTM(8),
                Dense(32, activation="relu"),
                Dense(6, activation="sigmoid"),
                Dropout(dropout),
            ]
        )
    else:
        model = Sequential(
            [
                Input(shape=(1,), dtype=string),
                layer_tokens,
                Embedding(input_dim=layer_tokens.vocabulary_size(), output_dim=32),
                Bidirectional(LSTM(8)),
                Dense(32, activation="relu"),
                Dense(6, activation="sigmoid"),
                Dropout(dropout),
            ]
        )
    return model


bidirecional = 0

layer_tokens = tokens(100, dados_treino)

# dropout = 0
model_uni_0 = gerar_modelo(0, layer_tokens, bidirecional)
model_uni_0.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model_uni_0.fit(
    dados_treino,
    labels_treino,
    epochs=50,
    batch_size=256,
    validation_data=(dados_validacao, labels_validacao),
    callbacks=[save_metrics_callback],
)

# dropout = .25
model_uni_25 = gerar_modelo(0.25, layer_tokens, bidirecional)
model_uni_25.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model_uni_25.fit(
    dados_treino,
    labels_treino,
    epochs=50,
    batch_size=256,
    validation_data=(dados_validacao, labels_validacao),
    callbacks=[save_metrics_callback],
)
# dropout = 0.5
model_uni_50 = gerar_modelo(0.5, layer_tokens, bidirecional)
model_uni_50.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model_uni_50.fit(
    dados_treino,
    labels_treino,
    epochs=50,
    batch_size=256,
    validation_data=(dados_validacao, labels_validacao),
    callbacks=[save_metrics_callback],
)


test_loss_uni_0, test_accuracy_uni_0 = model_uni_0.evaluate(dados_teste, labels_teste)
test_loss_uni_25, test_accuracy_uni_25 = model_uni_25.evaluate(
    dados_teste, labels_teste
)
test_loss_uni_25, test_accuracy_uni_25 = model_uni_25.evaluate(
    dados_teste, labels_teste
)
test_loss_uni_50, test_accuracy_uni_50 = model_uni_50.evaluate(
    dados_teste, labels_teste
)
test_loss_uni_50, test_accuracy_uni_50 = model_uni_50.evaluate(
    dados_teste, labels_teste
)


# Modelo com encoder bidirecional

bidirecional = 1

# dropout = 0
model_bi_0 = gerar_modelo(0, layer_tokens, bidirecional)
model_bi_0.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model_bi_0.fit(
    dados_treino,
    labels_treino,
    epochs=20,
    batch_size=256,
    validation_data=(dados_validacao, labels_validacao),
    callbacks=[save_metrics_callback],
)

# dropout = .25
model_bi_25 = gerar_modelo(0.25, layer_tokens, bidirecional)
model_bi_25.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model_bi_25.fit(
    dados_treino,
    labels_treino,
    epochs=20,
    batch_size=256,
    validation_data=(dados_validacao, labels_validacao),
    callbacks=[save_metrics_callback],
)
# dropout = 0.5
model_bi_50 = gerar_modelo(0.5, layer_tokens, bidirecional)
model_bi_50.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model_bi_50.fit(
    dados_treino,
    labels_treino,
    epochs=20,
    batch_size=256,
    validation_data=(dados_validacao, labels_validacao),
    callbacks=[save_metrics_callback],
)


test_loss_bi_0, test_accuracy_bi_0 = model_bi_0.evaluate(dados_teste, labels_teste)
test_loss_bi_25, test_accuracy_bi_25 = model_bi_25.evaluate(dados_teste, labels_teste)
test_loss_bi_25, test_accuracy_bi_25 = model_bi_25.evaluate(dados_teste, labels_teste)
test_loss_bi_50, test_accuracy_bi_50 = model_bi_50.evaluate(dados_teste, labels_teste)
test_loss_bi_50, test_accuracy_bi_50 = model_bi_50.evaluate(dados_teste, labels_teste)


test_results = [
    {"Test Accuracy": test_accuracy_uni_0, "Encoder": "unidirecional", "Dropout": 0},
    {
        "Test Accuracy": test_accuracy_uni_25,
        "Encoder": "unidirecional",
        "Dropout": 0.25,
    },
    {
        "Test Accuracy": test_accuracy_uni_50,
        "Encoder": "unidirecional",
        "Dropout": 0.50,
    },
    {"Test Accuracy": test_accuracy_bi_0, "Encoder": "bidirecional", "Dropout": 0},
    {"Test Accuracy": test_accuracy_bi_25, "Encoder": "bidirecional", "Dropout": 0.25},
    {"Test Accuracy": test_accuracy_bi_50, "Encoder": "bidirecional", "Dropout": 0.5},
]

df = pd.DataFrame(test_results)
df.to_csv("test_results.csv", index=False)
