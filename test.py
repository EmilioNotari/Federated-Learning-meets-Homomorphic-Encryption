import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tqdm import tqdm
from openfhe_lib.ckks.openFHE import * 


# =========================
#   CONSTANTES GLOBALES
# =========================

INPUT_DIM         = 31
LEARNING_RATE     = 0.01
EPOCHS            = 10000
LOCAL_ITERS       = 10
FED_ITERS         = 1000

DATASET_MAIN      = "data/breast-cancer.csv"
DATASET_TEST      = "data/test.csv"
DATASET_CLIENTS   = [
    ("Hospital1", "data/dataset1.csv", "/enc_weight_client1.txt"),
    ("Hospital2", "data/dataset2.csv", "/enc_weight_client2.txt"),
    ("Hospital3", "data/dataset3.csv", "/enc_weight_client3.txt"),
    ("Hospital4", "data/dataset4.csv", "/enc_weight_client4.txt"),
]

CSV_TRAIN_LOSS      = "training_loss.csv"
CSV_TRAIN_ACCURACY  = "training_accuracy.csv"
CSV_TEST_LOSS       = "test_loss.csv"
CSV_TEST_ACCURACY   = "test_accuracy.csv"


# =========================
#   DATASET HELPERS
# =========================

def swap_columns(df, col1, col2):
    col_list = list(df.columns)
    x, y = col_list.index(col1), col_list.index(col2)
    col_list[y], col_list[x] = col_list[x], col_list[y]
    return df[col_list]

def scale_dataset(df, overSample=False):
    X = df[df.columns[:-1]].values
    Y = df[df.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if overSample:
        ros = RandomOverSampler()
        X, Y = ros.fit_resample(X, Y)

    data = np.hstack((X, np.reshape(Y, (-1, 1))))
    X_train_tensor = torch.tensor(X, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y, dtype=torch.float32)
    data_tensor    = torch.tensor(data, dtype=torch.float32)

    return data_tensor, X_train_tensor, Y_train_tensor


# =========================
#   LOGISTIC REGRESSION
# =========================

class LogisticRegression(nn.Module):
    def __init__(self, num_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_features, 1)
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# =========================
#   TRAINING FUNCTIONS
# =========================

def decide(y):
    return 1. if y >= 0.5 else 0.

decide_vectorized = np.vectorize(decide)

def compute_accuracy(model, input, output):
    prediction = model(input).data.numpy()[:, 0]
    n_samples = prediction.shape[0] + 0.
    prediction = decide_vectorized(prediction)
    equal = prediction == output.data.numpy()
    return 100. * equal.sum() / n_samples

def compute_loss(model, X, Y):
    criterion = nn.BCELoss(reduction='mean')
    prediction = model(X)
    return criterion(prediction.squeeze(), Y).item()

def Training(X_train, Y_train, X_test, Y_test, input_dim=INPUT_DIM, epochs=EPOCHS, learning_rate=LEARNING_RATE, debug=True):
    model = LogisticRegression(input_dim)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

    losses = []
    accuracies = []

    for epoch in tqdm(range(epochs)):  
        optimizer.zero_grad()
        prediction = model(X_train)
        loss = criterion(prediction.squeeze(), Y_train)
        loss.backward()
        optimizer.step()

        train_acc = compute_accuracy(model, X_train, Y_train)
        losses.append(loss.item())
        accuracies.append(train_acc)

        if debug and (epoch + 1) % 50 == 0:
            print(f"[LOG] Epoch: {epoch+1:05d} | ACC: {train_acc:.2f}% | Loss: {loss.item():.3f}")

    return model, [accuracies, losses]


# =========================
#   MAIN
# =========================

if __name__ == "__main__":
    # Dataset inicial
    df = pd.read_csv(DATASET_MAIN)
    df = swap_columns(df, 'diagnosis', 'fractal_dimension_worst')
    df["diagnosis"] = (df["diagnosis"] == "M").astype(int)
    df_train, df_test = np.split(df.sample(frac=1), [int(0.8 * len(df))])
    train, X_train, Y_train = scale_dataset(df_train, True)
    test , X_test , Y_test  = scale_dataset(df_test , False)

    # Entrenamiento simple
    final_model, recorded = Training(X_train, Y_train, X_test, Y_test, INPUT_DIM, EPOCHS, LEARNING_RATE)
    train_accs, train_losses = recorded

    # =========================
    #   GUARDAR TRAINING
    # =========================
    pd.DataFrame({
        "epoch": list(range(1, len(train_losses)+1)),
        "loss": train_losses
    }).to_csv(CSV_TRAIN_LOSS, index=False)

    pd.DataFrame({
        "epoch": list(range(1, len(train_accs)+1)),
        "accuracy": train_accs
    }).to_csv(CSV_TRAIN_ACCURACY, index=False)

    # =========================
    #   GUARDAR TEST
    # =========================
    test_loss = compute_loss(final_model, X_test, Y_test)
    test_acc  = compute_accuracy(final_model, X_test, Y_test)

    pd.DataFrame({
        "phase": ["test"],
        "loss": [test_loss]
    }).to_csv(CSV_TEST_LOSS, index=False)

    pd.DataFrame({
        "phase": ["test"],
        "accuracy": [test_acc]
    }).to_csv(CSV_TEST_ACCURACY, index=False)

    print("[+] Resultados guardados en CSV")
