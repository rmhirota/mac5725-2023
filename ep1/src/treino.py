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


def gerar_modelo(tam, dropout, treino, layer_tokens, bidirecional):
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


def print_metrics(epoch, logs, tammax, batch_size, dropout, bidirecional):
    loss = logs["loss"]
    accuracy = logs["accuracy"]
    val_loss = logs["val_loss"]
    val_accuracy = logs["val_accuracy"]

    # Print metrics
    print(
        f"Epoch {epoch + 1} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f} - Val Loss: {val_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - Tammax: {tammax} - Batch size: {batch_size} - Dropout: {dropout:.2f}"
    )

    # Save metrics to a file
    with open("metrics_dropout.csv", "a") as file:
        file.write(
            f"Epoch {epoch + 1}, Loss {loss:.4f}, Accuracy {accuracy:.4f}, Val Loss {val_loss:.4f}, Val Accuracy {val_accuracy:.4f}, Tammax {tammax}, Batch size {batch_size}, Dropout {dropout:.2f}, Bidirecional {bidirecional}\n"
        )


def train_and_save_metrics(
    train_data,
    train_labels,
    validation_data,
    validation_labels,
    tammax,
    batch_size,
    dropout,
    bidirecional,
):
    layer_tokens = tokens(tammax, train_data)
    model = gerar_modelo(tammax, dropout, train_data, layer_tokens, bidirecional)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    save_metrics_callback = LambdaCallback(
        on_epoch_end=lambda epoch, logs: print_metrics(
            epoch, logs, tammax, batch_size, dropout, bidirecional
        )
        if (epoch + 1) % 5 == 0
        else None
    )

    model.fit(
        train_data,
        train_labels,
        epochs=100,
        batch_size=batch_size,
        validation_data=(validation_data, validation_labels),
        callbacks=[save_metrics_callback],
    )


hyperparameter_combinations = [
    {"tammax": 100, "batch_size": 256, "dropout": 0},
    {"tammax": 200, "batch_size": 256, "dropout": 0},
    {"tammax": 100, "batch_size": 512, "dropout": 0},
    {"tammax": 200, "batch_size": 512, "dropout": 0},
    {"tammax": 100, "batch_size": 256, "dropout": 0.25},
    {"tammax": 200, "batch_size": 256, "dropout": 0.25},
    {"tammax": 100, "batch_size": 512, "dropout": 0.25},
    {"tammax": 200, "batch_size": 512, "dropout": 0.25},
    {"tammax": 100, "batch_size": 256, "dropout": 0.5},
    {"tammax": 200, "batch_size": 256, "dropout": 0.5},
    {"tammax": 100, "batch_size": 512, "dropout": 0.5},
    {"tammax": 200, "batch_size": 512, "dropout": 0.5},
]

dados_treino = ler_dados("base_treino.csv")
dados_validacao = ler_dados("base_validacao.csv")
dados_teste = ler_dados("base_teste.csv")

labels_treino = ler_labels("base_treino.csv")
labels_validacao = ler_labels("base_validacao.csv")
labels_teste = ler_labels("base_teste.csv")

for model_index, hyperparameters in enumerate(hyperparameter_combinations, start=1):
    train_and_save_metrics(
        dados_treino,
        labels_treino,
        dados_validacao,
        labels_validacao,
        hyperparameters["tammax"],
        hyperparameters["batch_size"],
        hyperparameters["dropout"],
        0,
    )


# Bidirecional ----

for model_index, hyperparameters in enumerate(hyperparameter_combinations, start=1):
    train_and_save_metrics(
        dados_treino,
        labels_treino,
        dados_validacao,
        labels_validacao,
        hyperparameters["tammax"],
        hyperparameters["batch_size"],
        hyperparameters["dropout"],
        1,
    )
