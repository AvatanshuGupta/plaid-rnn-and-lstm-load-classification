import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os
FILE = 'features/all_features.csv'
TARGET = 'load'
SEQ_LEN = 30
HIDDEN = 64
EPOCHS = 100
BATCH_SIZE = 64
LR = 0.001
np.random.seed(42)
os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_plots', exist_ok=True)

def load_data(file):
    df = pd.read_csv(file)
    features = [c for c in df.select_dtypes(include=np.number).columns if c not in ['window_idx']]
    if 'id' in features:
        features.remove('id')
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    X = {}
    y = {}
    for rid, g in df.groupby('id'):
        g = g.sort_values('window_idx')
        X[rid] = g[features].to_numpy(dtype=np.float32)
        y[rid] = g[TARGET].iloc[0]
    return (X, y, features)

def preprocess(X, train_ids):
    train_data = np.vstack([X[i] for i in train_ids])
    mean = np.nanmean(train_data, axis=0)
    std = np.nanstd(train_data, axis=0) + 1e-08
    for i in X:
        X[i] = np.nan_to_num(X[i], nan=mean)
        X[i] = (X[i] - mean) / std
    return (X, mean, std)

def make_chunks(X, ids, labels, seq_len):
    data = []
    target = []
    records = []
    for rid in ids:
        sequence = X[rid]
        if len(sequence) < seq_len:
            sequence = np.vstack([sequence, np.repeat(sequence[-1:], seq_len - len(sequence), axis=0)])
        for start in range(0, len(sequence) - seq_len + 1, seq_len):
            chunk = sequence[start:start + seq_len]
            data.append(chunk)
            target.append(labels[rid])
            records.append(rid)
    return (np.array(data), np.array(target), np.array(records))

class ScratchRNN:

    def __init__(self, input_size, hidden_size, output_size):
        rng = np.random.default_rng(42)
        self.Wx1 = rng.normal(0, 0.05, (input_size, hidden_size))
        self.Wh1 = rng.normal(0, 0.05, (hidden_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.Wx2 = rng.normal(0, 0.05, (hidden_size, hidden_size))
        self.Wh2 = rng.normal(0, 0.05, (hidden_size, hidden_size))
        self.b2 = np.zeros(hidden_size)
        self.W3 = rng.normal(0, 0.05, (2 * hidden_size, hidden_size))
        self.b3 = np.zeros(hidden_size)
        self.W4 = rng.normal(0, 0.05, (hidden_size, output_size))
        self.b4 = np.zeros(output_size)
        self.params = [self.Wx1, self.Wh1, self.b1, self.Wx2, self.Wh2, self.b2, self.W3, self.b3, self.W4, self.b4]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step = 0

    def forward(self, X):
        B, T, F = X.shape
        h1 = np.zeros((B, self.Wh1.shape[0]))
        h2 = np.zeros((B, self.Wh2.shape[0]))
        h1_list = []
        h2_list = []
        for t in range(T):
            h1 = np.tanh(X[:, t] @ self.Wx1 + h1 @ self.Wh1 + self.b1)
            h2 = np.tanh(h1 @ self.Wx2 + h2 @ self.Wh2 + self.b2)
            h1_list.append(h1.copy())
            h2_list.append(h2.copy())
        h_mean = np.mean(h2_list, axis=0)
        h_last = h2
        z = np.concatenate([h_mean, h_last], axis=1)
        a = z @ self.W3 + self.b3
        a = np.maximum(0, a)
        logits = a @ self.W4 + self.b4
        cache = (h1_list, h2_list, z, a)
        return (logits, cache)

    def backward(self, X, y, logits, cache):
        h1_list, h2_list, z, a = cache
        B, T, F = X.shape
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits_shifted)
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        dlogits = probs.copy()
        dlogits[np.arange(B), y] -= 1
        dlogits /= B
        dW4 = a.T @ dlogits
        db4 = np.sum(dlogits, axis=0)
        da = dlogits @ self.W4.T
        da[a <= 0] = 0
        dW3 = z.T @ da
        db3 = np.sum(da, axis=0)
        dz = da @ self.W3.T
        dh_mean = dz[:, :self.Wh2.shape[0]]
        dh2 = dz[:, self.Wh2.shape[0]:]
        dh_mean /= T
        dWx1 = np.zeros_like(self.Wx1)
        dWh1 = np.zeros_like(self.Wh1)
        db1 = np.zeros_like(self.b1)
        dWx2 = np.zeros_like(self.Wx2)
        dWh2 = np.zeros_like(self.Wh2)
        db2 = np.zeros_like(self.b2)
        dh1_next = np.zeros_like(h1_list[0])
        for t in range(T - 1, -1, -1):
            h2 = h2_list[t]
            h1 = h1_list[t]
            dh2_total = dh2 + dh_mean
            dh2_raw = dh2_total * (1 - h2 * h2)
            dWx2 += h1.T @ dh2_raw
            db2 += np.sum(dh2_raw, axis=0)
            if t > 0:
                dWh2 += h2_list[t - 1].T @ dh2_raw
            dh1 = dh2_raw @ self.Wx2.T
            dh1 += dh1_next
            dh1_raw = dh1 * (1 - h1 * h1)
            dWx1 += X[:, t].T @ dh1_raw
            db1 += np.sum(dh1_raw, axis=0)
            if t > 0:
                dWh1 += h1_list[t - 1].T @ dh1_raw
            dh1_next = dh1_raw @ self.Wh1.T
            dh2 = dh2_raw @ self.Wh2.T
        grads = [dWx1, dWh1, db1, dWx2, dWh2, db2, dW3, db3, dW4, db4]
        for g in grads:
            np.clip(g, -5, 5, out=g)
        return grads

    def update(self, grads, lr=0.001):
        self.step += 1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-08
        for i in range(len(self.params)):
            p = self.params[i]
            g = grads[i]
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * g * g
            m_hat = self.m[i] / (1 - beta1 ** self.step)
            v_hat = self.v[i] / (1 - beta2 ** self.step)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

def cross_entropy(logits, y):
    x = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(x)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    return -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-08))

def train(model, X, y, epochs=100, batch_size=64, lr=0.001):
    loss_history = []
    for epoch in range(epochs):
        order = np.random.permutation(len(X))
        total_loss = 0
        for i in range(0, len(X), batch_size):
            idx = order[i:i + batch_size]
            xb = X[idx]
            yb = y[idx]
            logits, cache = model.forward(xb)
            loss = cross_entropy(logits, yb)
            grads = model.backward(xb, yb, logits, cache)
            model.update(grads, lr)
            total_loss += loss * len(xb)
        avg_loss = total_loss / len(X)
        loss_history.append(avg_loss)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1:3d} | Loss: {avg_loss:.4f}')
    return loss_history

def predict_proba(model, X):
    logits, _ = model.forward(X)
    x = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def record_prediction(probs, y, records):
    unique_records = np.unique(records)
    record_probs = []
    record_y = []
    for rid in unique_records:
        mask = records == rid
        record_probs.append(probs[mask].mean(axis=0))
        record_y.append(y[mask][0])
    record_probs = np.array(record_probs)
    record_y = np.array(record_y)
    pred = record_probs.argmax(axis=1)
    return (record_y, pred, record_probs)

def shap_analysis(model, X_train, X_test, features):
    try:
        import shap
    except ImportError:
        print('\nInstall SHAP using:')
        print('pip install shap')
        return
    background = X_train[:20]
    test_sample = X_test[:5]
    background_flat = background.mean(axis=1)
    test_flat = test_sample.mean(axis=1)

    def model_wrapper(X):
        X_seq = np.repeat(X[:, None, :], X_train.shape[1], axis=1)
        return predict_proba(model, X_seq)
    print('\nRunning SHAP...')
    explainer = shap.KernelExplainer(model_wrapper, background_flat)
    shap_values = explainer.shap_values(test_flat, nsamples=100)
    if isinstance(shap_values, list):
        values = np.array(shap_values)
    else:
        values = np.array(shap_values)
    if values.ndim == 3:
        importance = np.mean(np.abs(values), axis=(0, 1))
    else:
        importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(importance)[::-1]
    print('\nSHAP Feature Importance:')
    for i in order:
        print(f'{features[i]:25s} {importance[i]:.6f}')
    plt.figure(figsize=(8, 5))
    plt.barh(np.array(features)[order], importance[order])
    plt.xlabel('Mean |SHAP value|')
    plt.title('Feature Importance for Load Classification')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('saved_plots/shap_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
print('\nLoading data...')
df = pd.read_csv(FILE)
print(df[TARGET].value_counts())
print('\nUnique classes:')
print(df[TARGET].unique())
X, labels, features = load_data(FILE)
print('\nFeatures:')
print(features)
print('\nLoad classes:')
print(sorted(set(labels.values())))
classes = sorted(set(labels.values()))
class_to_id = {c: i for i, c in enumerate(classes)}
labels = {rid: class_to_id[label] for rid, label in labels.items()}
ids = np.array(list(X.keys()))
record_y = np.array([labels[rid] for rid in ids])
train_ids, test_ids = train_test_split(ids, test_size=0.2, stratify=record_y, random_state=42)
X, train_mean, train_std = preprocess(X, train_ids)
X_train, y_train, train_records = make_chunks(X, train_ids, labels, SEQ_LEN)
X_test, y_test, test_records = make_chunks(X, test_ids, labels, SEQ_LEN)
print('\nTrain shape:', X_train.shape)
print('Test shape:', X_test.shape)
model = ScratchRNN(input_size=X_train.shape[2], hidden_size=HIDDEN, output_size=len(classes))
print('\nTraining Scratch RNN...')
loss_history = train(model, X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR)
np.savez('saved_models/scratch_rnn_model.npz', Wx1=model.Wx1, Wh1=model.Wh1, b1=model.b1, Wx2=model.Wx2, Wh2=model.Wh2, b2=model.b2, W3=model.W3, b3=model.b3, W4=model.W4, b4=model.b4, train_mean=train_mean, train_std=train_std, classes=np.array(classes), features=np.array(features))
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(loss_history) + 1), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Scratch RNN Training Loss')
plt.tight_layout()
plt.savefig('saved_plots/training_loss.png', dpi=300, bbox_inches='tight')
plt.close()
probs = predict_proba(model, X_test)
Y, pred, record_probs = record_prediction(probs, y_test, test_records)
accuracy = accuracy_score(Y, pred)
f1 = f1_score(Y, pred, average='macro')
print(f'Accuracy: {accuracy:.4f}')
print(f'Macro F1: {f1:.4f}')
print('\nConfusion Matrix:')
print(confusion_matrix(Y, pred))
print('\nClasses:')
print(classes)
cm = confusion_matrix(Y, pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.xticks(range(len(classes)), classes)
plt.yticks(range(len(classes)), classes)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center')
plt.tight_layout()
plt.savefig('saved_plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
shap_analysis(model, X_train, X_test, features)
