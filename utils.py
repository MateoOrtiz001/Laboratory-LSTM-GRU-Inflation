import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate=0.001,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Actualiza parámetros usando Adam

    Arguments:
    parameters -- dict con llaves: 'Wf', 'bf', 'Wi', ...
    grads -- dict con llaves: 'dWf', 'dbf', 'dWi', ...  (con prefijo 'd')
    """
    v_corrected = {}
    s_corrected = {}

    for key in parameters.keys():
        grad_key = 'd' + key

        if grad_key in grads:
            # Inicializa v/s si no existen
            if key not in v:
                v[key] = np.zeros_like(parameters[key])
            if key not in s:
                s[key] = np.zeros_like(parameters[key])

            # Usa grads[grad_key] en vez de grads[key]
            v[key] = beta1 * v[key] + (1 - beta1) * grads[grad_key]
            s[key] = beta2 * s[key] + (1 - beta2) * (grads[grad_key] ** 2)

            v_corrected[key] = v[key] / (1 - beta1 ** t)
            s_corrected[key] = s[key] / (1 - beta2 ** t)

            parameters[key] -= learning_rate * (v_corrected[key] / (np.sqrt(s_corrected[key]) + epsilon))

    return parameters, v, s


def compute_cost(y_pred, y_true):
    """
    Calcula el costo MSE (Mean Squared Error)
    """
    m = y_true.shape[1] # Number of examples (batch size)
    cost = np.sum((y_pred - y_true) ** 2) / (2 * m)
    return np.squeeze(cost)

def clip_gradients(gradients, maxValue):
    for key in gradients:
        if isinstance(gradients[key], np.ndarray):
            np.clip(gradients[key], -maxValue, maxValue, out=gradients[key])
    return gradients


def plot_training(costs):
    """
    Grafica la evolución del costo durante el entrenamiento

    Arguments:
    costs -- lista de costos por época
    """
    plt.figure(figsize=(8, 6))
    plt.plot(costs, linewidth=2)
    plt.xlabel('Época', fontsize=12)
    plt.ylabel('Costo (MSE)', fontsize=12)
    plt.title('LSTM', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    """
    Crea minibatches aleatorios a partir de (X, Y)
    X -- numpy array de forma (n_x, m, T_x)
    Y -- numpy array de forma (n_y, m, 1)
    """
    np.random.seed(seed)
    m = X.shape[1]  # número de ejemplos
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation, :]
    shuffled_Y = Y[:, permutation, :]

    mini_batches = []
    num_complete_minibatches = m // mini_batch_size

    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    # Si sobran ejemplos
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:, :]
        mini_batches.append((mini_batch_X, mini_batch_Y))

    return mini_batches

def reshape_for_lstm(X, y):
    """
    Transforma datos al formato requerido por tu implementación LSTM

    LSTM espera:
    - X: shape (n_x, m, T_x) donde:
        n_x = número de features (en este caso 1: solo inflación)
        m = batch size (número de muestras)
        T_x = longitud de secuencia temporal
    - y: shape (n_y, m, 1) donde:
        n_y = número de salidas (1: inflación)
        m = batch size

    Argumentos:
    X -- array shape (m, T_x)
    y -- array shape (m,)

    Retorna:
    X_reshaped -- array shape (n_x, m, T_x)
    y_reshaped -- array shape (n_y, m, 1)
    """
    # X: de (m, T_x) a (n_x, m, T_x) -> (1, m, T_x)
    # Assuming n_x = 1 feature (Inflation)
    X_reshaped = X.reshape(1, X.shape[0], X.shape[1])


    # y: de (m,) a (n_y, m, 1) -> (1, m, 1)
    # Assuming n_y = 1 output (Inflation)
    y_reshaped = y.reshape(1, y.shape[0], 1)

    return X_reshaped, y_reshaped