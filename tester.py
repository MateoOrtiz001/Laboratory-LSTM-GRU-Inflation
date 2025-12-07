import numpy as np
import random
from utils import sigmoid


def create_sequences_test(create_sequences):
    # Tester de la funcion create_sequences
    # Fijamos una semilla para los test
    np.random.seed(42)
    # Generamos 20 datos aleatorios
    random_data = np.random.rand(20)

    # Imprimimos los datos generados
    print("Datos aleatorios generados:")
    print(random_data)

    # Definimos los parámetros
    n_steps_in = [5,3,6,3]
    n_steps_out = [2,1,4,3]

    # Iteramos sobre los parametros
    for n_steps_in_value, n_steps_out_value in zip(n_steps_in, n_steps_out):
        X, Y = create_sequences(random_data, n_steps_in_value, n_steps_out_value)
        # Imprimimos solo el primer caso
        if n_steps_in_value == 5:
            print("\nPrimer secuencia de entrada (X):")
            print(X[0])
            print("\nPrimer secuencia de salida (Y):")
            print(Y[0])
        # Verificamos el número de secuencias esperadas
        expected_num_sequences = len(random_data) - n_steps_in_value - n_steps_out_value + 1
        assert len(X) == expected_num_sequences, f"Expected {expected_num_sequences} sequences, but got {len(X)}"
        assert len(Y) == expected_num_sequences, f"Expected {expected_num_sequences} sequences, but got {len(Y)}"
        # Verificamos el shape de las secuencias
        assert X.shape == (expected_num_sequences, n_steps_in_value), f"Expected X shape {(expected_num_sequences, n_steps_in_value)}, but got {X.shape}"
        assert Y.shape == (expected_num_sequences, n_steps_out_value), f"Expected Y shape {(expected_num_sequences, n_steps_out_value)}, but got {Y.shape}"
        # Verificamos el primer elemento de las secuencias
        assert np.array_equal(X[0], random_data[:n_steps_in_value]), f"First input sequence is incorrect. Expected {random_data[:n_steps_in_value]}, but got {X[0]}"
        assert np.array_equal(Y[0], random_data[n_steps_in_value:n_steps_in_value + n_steps_out_value]), f"First output sequence is incorrect. Expected {random_data[n_steps_in_value:n_steps_in_value + n_steps_out_value]}, but got {Y[0]}"

    print('\033[92mTest Aprobado\033[0m')

def reshape_data_test(reshape_data):
    # Tester de la funcion reshape_for_lstm
    # Fijamos una semilla para los test para reproducibilidad
    np.random.seed(42)

    # Definimos diferentes tamaños de ventana y pasos de salida para probar
    test_params = [
        (10, 3),  # T_x=10, n_steps_out=3
        (5, 1),   # T_x=5, n_steps_out=1
        (15, 5)   # T_x=15, n_steps_out=5
    ]

    for n_steps_in, n_steps_out in test_params:
        # Generamos datos de ejemplo con un número arbitrario de muestras (m)
        m = 25 # Número de muestras
        X_original = np.random.rand(m, n_steps_in)
        Y_original = np.random.rand(m, n_steps_out)

        # Aplicamos la función de reshape
        X_reshaped, Y_reshaped = reshape_data(X_original, Y_original)

        # Imprimimos las formas originales y transformadas para el primer caso
        if n_steps_in == 10:
            print(f"Probando con n_steps_in={n_steps_in}, n_steps_out={n_steps_out}, m={m}")
            print("Shape inicial de X:", X_original.shape)
            print("Reshape de X:", X_reshaped.shape)
            print("Shape inicial de Y:", Y_original.shape)
            print("Reshape de Y:", Y_reshaped.shape)
            print("-" * 20)


        # Verificamos las formas transformadas
        expected_X_shape = (1, m, n_steps_in)
        expected_Y_shape = (n_steps_out, m, 1)

        assert X_reshaped.shape == expected_X_shape, f"Expected X shape {expected_X_shape}, but got {X_reshaped.shape}"
        assert Y_reshaped.shape == expected_Y_shape, f"Expected Y shape {expected_Y_shape}, but got {Y_reshaped.shape}"

        # Verificamos el contenido (comparando elementos seleccionados)
        # X_reshaped[0, :, i] debería ser igual a X_original[:, i]
        for i in range(n_steps_in):
            assert np.array_equal(X_reshaped[0, :, i], X_original[:, i]), f"Content mismatch in X at step {i}"

        # Y_reshaped[i, :, 0] debería ser igual a Y_original[:, i]
        for i in range(n_steps_out):
            assert np.array_equal(Y_reshaped[i, :, 0], Y_original[:, i]), f"Content mismatch in Y at step {i}"


    print('\033[92mTest Aprobado\033[0m')


def lstm_cell_forward_test(lstm_cell_forward):
    np.random.seed(7)
    n_x, n_h, m = 3, 2, 1
    xt = np.random.randn(n_x, m)
    h_prev = np.random.randn(n_h, m)
    c_prev = np.random.randn(n_h, m)

    parameters = {
        "Wf": np.random.randn(n_h, n_h + n_x),
        "bf": np.random.randn(n_h, 1),
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wc": np.random.randn(n_h, n_h + n_x),
        "bc": np.random.randn(n_h, 1),
        "Wo": np.random.randn(n_h, n_h + n_x),
        "bo": np.random.randn(n_h, 1),
    }

    h_next, c_next, cache = lstm_cell_forward(xt, h_prev, c_prev, parameters)

    concat = np.concatenate((h_prev, xt))
    ft_exp = sigmoid(np.dot(parameters["Wf"], concat) + parameters["bf"])
    ut_exp = sigmoid(np.dot(parameters["Wu"], concat) + parameters["bu"])
    cct_exp = np.tanh(np.dot(parameters["Wc"], concat) + parameters["bc"])
    c_next_exp = ft_exp * c_prev + ut_exp * cct_exp
    ot_exp = sigmoid(np.dot(parameters["Wo"], concat) + parameters["bo"])
    h_next_exp = ot_exp * np.tanh(c_next_exp)

    assert h_next.shape == (n_h, m)
    assert c_next.shape == (n_h, m)
    assert np.allclose(h_next, h_next_exp)
    assert np.allclose(c_next, c_next_exp)
    assert len(cache) == 10
    print('\033[92mTest Aprobado\033[0m')


def lstm_forward_test(lstm_forward):
    np.random.seed(8)
    n_x, m, T_x, n_h, n_y = 2, 3, 4, 3, 1
    x = np.random.randn(n_x, m, T_x)
    h0 = np.random.randn(n_h, m)

    parameters = {
        "Wf": np.random.randn(n_h, n_h + n_x),
        "bf": np.random.randn(n_h, 1),
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wc": np.random.randn(n_h, n_h + n_x),
        "bc": np.random.randn(n_h, 1),
        "Wo": np.random.randn(n_h, n_h + n_x),
        "bo": np.random.randn(n_h, 1),
        "Wy": np.random.randn(n_y, n_h),
        "by": np.random.randn(n_y, 1),
    }

    def manual_cell(xt, h_prev, c_prev):
        concat = np.concatenate((h_prev, xt))
        ft = sigmoid(np.dot(parameters["Wf"], concat) + parameters["bf"])
        ut = sigmoid(np.dot(parameters["Wu"], concat) + parameters["bu"])
        cct = np.tanh(np.dot(parameters["Wc"], concat) + parameters["bc"])
        c_next = ft * c_prev + ut * cct
        ot = sigmoid(np.dot(parameters["Wo"], concat) + parameters["bo"])
        h_next = ot * np.tanh(c_next)
        return h_next, c_next

    h, y, c, caches = lstm_forward(x, h0, parameters)

    h_exp = np.zeros((n_h, m, T_x))
    c_exp = np.zeros((n_h, m, T_x))
    h_prev = h0
    c_prev = np.zeros((n_h, m))
    for t in range(T_x):
        h_prev, c_prev = manual_cell(x[:, :, t], h_prev, c_prev)
        h_exp[:, :, t] = h_prev
        c_exp[:, :, t] = c_prev
    y_exp = np.dot(parameters['Wy'], h_exp[:, :, -1]) + parameters['by']

    assert h.shape == (n_h, m, T_x)
    assert c.shape == (n_h, m, T_x)
    assert y.shape == (n_y, m)
    assert np.allclose(h, h_exp)
    assert np.allclose(c, c_exp)
    assert np.allclose(y, y_exp)
    assert isinstance(caches, tuple) and len(caches[0]) == T_x
    print('\033[92mTest Aprobado\033[0m')


def lstm_cell_backward_test(lstm_cell_backward, lstm_cell_forward):
    np.random.seed(9)
    n_x, n_h, m = 2, 2, 1
    xt = np.random.randn(n_x, m)
    h_prev = np.random.randn(n_h, m)
    c_prev = np.random.randn(n_h, m)
    parameters = {
        "Wf": np.random.randn(n_h, n_h + n_x),
        "bf": np.random.randn(n_h, 1),
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wc": np.random.randn(n_h, n_h + n_x),
        "bc": np.random.randn(n_h, 1),
        "Wo": np.random.randn(n_h, n_h + n_x),
        "bo": np.random.randn(n_h, 1),
    }

    h_next, c_next, cache = lstm_cell_forward(xt, h_prev, c_prev, parameters)
    dh_next = np.random.randn(n_h, m)
    dc_next = np.random.randn(n_h, m)
    grads = lstm_cell_backward(dh_next, dc_next, cache)

    # Finite differences on a single weight to spot-check gradients
    epsilon = 1e-7
    def loss_with_params(pmat):
        params_copy = parameters.copy()
        params_copy["Wf"] = pmat
        h_eps, c_eps, _ = lstm_cell_forward(xt, h_prev, c_prev, params_copy)
        return float(np.sum(h_eps * dh_next) + np.sum(c_eps * dc_next))

    w_original = parameters["Wf"][0, 0]
    W_plus = parameters["Wf"].copy()
    W_minus = parameters["Wf"].copy()
    W_plus[0, 0] = w_original + epsilon
    W_minus[0, 0] = w_original - epsilon
    grad_numeric = (loss_with_params(W_plus) - loss_with_params(W_minus)) / (2 * epsilon)
    grad_analytic = grads["dWf"][0, 0]

    assert grads["dxt"].shape == (n_x, m)
    assert grads["dh_prev"].shape == (n_h, m)
    assert grads["dc_prev"].shape == (n_h, m)
    assert np.allclose(grad_numeric, grad_analytic, atol=1e-5)
    print('\033[92mTest Aprobado\033[0m')


def lstm_backward_test(lstm_backward, lstm_cell_backward, lstm_cell_forward):
    np.random.seed(10)
    n_x, m, T_x, n_h = 2, 2, 3, 3
    x = np.random.randn(n_x, m, T_x)
    h0 = np.random.randn(n_h, m)
    parameters = {
        "Wf": np.random.randn(n_h, n_h + n_x),
        "bf": np.random.randn(n_h, 1),
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wc": np.random.randn(n_h, n_h + n_x),
        "bc": np.random.randn(n_h, 1),
        "Wo": np.random.randn(n_h, n_h + n_x),
        "bo": np.random.randn(n_h, 1),
    }

    # Forward to build caches
    caches_list = []
    h_prev = h0
    c_prev = np.zeros((n_h, m))
    for t in range(T_x):
        h_prev, c_prev, cache = lstm_cell_forward(x[:, :, t], h_prev, c_prev, parameters)
        caches_list.append(cache)
    caches = (caches_list, x)

    dh = np.random.randn(n_h, m, T_x)
    grads = lstm_backward(dh, caches, m)

    # Manual accumulation using lstm_cell_backward
    dx_exp = np.zeros((n_x, m, T_x))
    dh_prevt = np.zeros((n_h, m))
    dc_prevt = np.zeros((n_h, m))
    dWf_exp = np.zeros((n_h, n_h + n_x))
    dWu_exp = np.zeros((n_h, n_h + n_x))
    dWc_exp = np.zeros((n_h, n_h + n_x))
    dWo_exp = np.zeros((n_h, n_h + n_x))
    dbf_exp = np.zeros((n_h, 1))
    dbu_exp = np.zeros((n_h, 1))
    dbc_exp = np.zeros((n_h, 1))
    dbo_exp = np.zeros((n_h, 1))
    for t in reversed(range(T_x)):
        g = lstm_cell_backward(dh[:, :, t] + dh_prevt, dc_prevt, caches_list[t])
        dh_prevt = g["dh_prev"]
        dc_prevt = g["dc_prev"]
        dx_exp[:, :, t] = g["dxt"]
        dWf_exp += g["dWf"]
        dWu_exp += g["dWu"]
        dWc_exp += g["dWc"]
        dWo_exp += g["dWo"]
        dbf_exp += g["dbf"]
        dbu_exp += g["dbu"]
        dbc_exp += g["dbc"]
        dbo_exp += g["dbo"]
    dWf_exp /= m; dWu_exp /= m; dWc_exp /= m; dWo_exp /= m
    dbf_exp /= m; dbu_exp /= m; dbc_exp /= m; dbo_exp /= m

    assert np.allclose(grads["dx"], dx_exp)
    assert np.allclose(grads["dWf"], dWf_exp)
    assert np.allclose(grads["dWu"], dWu_exp)
    assert np.allclose(grads["dWc"], dWc_exp)
    assert np.allclose(grads["dWo"], dWo_exp)
    assert np.allclose(grads["dbf"], dbf_exp)
    assert np.allclose(grads["dbu"], dbu_exp)
    assert np.allclose(grads["dbc"], dbc_exp)
    assert np.allclose(grads["dbo"], dbo_exp)
    print('\033[92mTest Aprobado\033[0m')


def initialize_LSTM_parameters_test(initialize_LSTM_parameters):
    n_h, n_x, n_y = 4, 3, 2
    params = initialize_LSTM_parameters(n_h, n_x, n_y)
    expected_keys = {"Wf", "bf", "Wu", "bu", "Wc", "bc", "Wo", "bo", "Wy", "by"}
    assert set(params.keys()) == expected_keys
    assert params["Wf"].shape == (n_h, n_h + n_x)
    assert params["Wy"].shape == (n_y, n_h)
    std_expected = np.sqrt(2.0 / (n_h + n_x))
    assert np.isclose(params["Wf"].std(), std_expected, rtol=0.3)
    assert np.all(params["bf"] == 0)
    print('\033[92mTest Aprobado\033[0m')


def model_lstm_adam_test(model_lstm_adam):
    np.random.seed(11)
    n_x, m, T_x, n_y = 1, 4, 3, 1
    X = np.random.randn(n_x, m, T_x)
    Y = np.random.randn(n_y, m, 1)
    params = model_lstm_adam(X, Y, n_h=3, learning_rate=0.01, mini_batch_size=2,
                             num_epochs=2, print_cost=False)
    expected_keys = {"Wf", "bf", "Wu", "bu", "Wc", "bc", "Wo", "bo", "Wy", "by"}
    assert set(params.keys()) == expected_keys
    for v in params.values():
        assert np.isfinite(v).all()
    print('\033[92mTest Aprobado\033[0m')


def predict_lstm_test(predict_lstm):
    np.random.seed(12)
    n_x, m, T_x, n_h, n_y = 1, 2, 3, 2, 1
    x = np.random.randn(n_x, m, T_x)
    parameters = {
        "Wf": np.random.randn(n_h, n_h + n_x),
        "bf": np.random.randn(n_h, 1),
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wc": np.random.randn(n_h, n_h + n_x),
        "bc": np.random.randn(n_h, 1),
        "Wo": np.random.randn(n_h, n_h + n_x),
        "bo": np.random.randn(n_h, 1),
        "Wy": np.random.randn(n_y, n_h),
        "by": np.random.randn(n_y, 1),
    }

    # Manual forward
    h_next = np.zeros((n_h, m))
    c_next = np.zeros((n_h, m))
    for t in range(T_x):
        concat = np.concatenate((h_next, x[:, :, t]))
        ft = sigmoid(np.dot(parameters["Wf"], concat) + parameters["bf"])
        ut = sigmoid(np.dot(parameters["Wu"], concat) + parameters["bu"])
        cct = np.tanh(np.dot(parameters["Wc"], concat) + parameters["bc"])
        c_next = ft * c_next + ut * cct
        ot = sigmoid(np.dot(parameters["Wo"], concat) + parameters["bo"])
        h_next = ot * np.tanh(c_next)
    y_exp = np.dot(parameters['Wy'], h_next) + parameters['by']

    y_pred = predict_lstm(x, parameters)
    assert y_pred.shape == (n_y, m)
    assert np.allclose(y_pred, y_exp)
    print('\033[92mTest Aprobado\033[0m')


def gru_cell_forward_test(gru_cell_forward):
    np.random.seed(13)
    n_x, n_h, m = 2, 2, 1
    xt = np.random.randn(n_x, m)
    h_prev = np.random.randn(n_h, m)
    parameters = {
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wh": np.random.randn(n_h, n_h + n_x),
        "bh": np.random.randn(n_h, 1),
        "Wr": np.random.randn(n_h, n_h + n_x),
        "br": np.random.randn(n_h, 1),
    }

    h_next, cache = gru_cell_forward(xt, h_prev, parameters)

    concat = np.concatenate((h_prev, xt))
    ut = sigmoid(np.dot(parameters["Wu"], concat) + parameters["bu"])
    rt = sigmoid(np.dot(parameters["Wr"], concat) + parameters["br"])
    hht = np.tanh(np.dot(parameters["Wh"], np.concatenate((rt * h_prev, xt))) + parameters["bh"])
    h_exp = (1 - ut) * h_prev + ut * hht

    assert h_next.shape == (n_h, m)
    assert np.allclose(h_next, h_exp)
    assert len(cache) == 7
    print('\033[92mTest Aprobado\033[0m')


def gru_forward_test(gru_forward):
    np.random.seed(14)
    n_x, m, T_x, n_h, n_y = 2, 2, 3, 2, 1
    x = np.random.randn(n_x, m, T_x)
    h0 = np.random.randn(n_h, m)
    parameters = {
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wh": np.random.randn(n_h, n_h + n_x),
        "bh": np.random.randn(n_h, 1),
        "Wr": np.random.randn(n_h, n_h + n_x),
        "br": np.random.randn(n_h, 1),
        "Wy": np.random.randn(n_y, n_h),
        "by": np.random.randn(n_y, 1),
    }

    h, y, caches = gru_forward(x, h0, parameters)

    h_exp = np.zeros((n_h, m, T_x))
    h_prev = h0
    for t in range(T_x):
        concat = np.concatenate((h_prev, x[:, :, t]))
        ut = sigmoid(np.dot(parameters["Wu"], concat) + parameters["bu"])
        rt = sigmoid(np.dot(parameters["Wr"], concat) + parameters["br"])
        hht = np.tanh(np.dot(parameters["Wh"], np.concatenate((rt * h_prev, x[:, :, t]))) + parameters["bh"])
        h_prev = (1 - ut) * h_prev + ut * hht
        h_exp[:, :, t] = h_prev
    y_exp = np.dot(parameters['Wy'], h_exp[:, :, -1]) + parameters['by']

    assert h.shape == (n_h, m, T_x)
    assert y.shape == (n_y, m, 1)
    assert np.allclose(h, h_exp)
    assert np.allclose(y, y_exp)
    assert isinstance(caches, tuple) and len(caches[0]) == T_x
    print('\033[92mTest Aprobado\033[0m')


def gru_cell_backward_test(gru_cell_backward, gru_cell_forward):
    np.random.seed(15)
    n_x, n_h, m = 2, 2, 1
    xt = np.random.randn(n_x, m)
    h_prev = np.random.randn(n_h, m)
    parameters = {
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wh": np.random.randn(n_h, n_h + n_x),
        "bh": np.random.randn(n_h, 1),
        "Wr": np.random.randn(n_h, n_h + n_x),
        "br": np.random.randn(n_h, 1),
    }

    h_next, cache = gru_cell_forward(xt, h_prev, parameters)
    dh_next = np.random.randn(n_h, m)
    grads = gru_cell_backward(dh_next, cache)

    epsilon = 1e-7
    def loss_with_params(pmat):
        params_copy = parameters.copy()
        params_copy["Wu"] = pmat
        h_eps, _ = gru_cell_forward(xt, h_prev, params_copy)
        return float(np.sum(h_eps * dh_next))

    w_orig = parameters["Wu"][0, 0]
    W_plus = parameters["Wu"].copy()
    W_minus = parameters["Wu"].copy()
    W_plus[0, 0] = w_orig + epsilon
    W_minus[0, 0] = w_orig - epsilon
    grad_numeric = (loss_with_params(W_plus) - loss_with_params(W_minus)) / (2 * epsilon)
    grad_analytic = grads["dWu"][0, 0]

    assert grads["dxt"].shape == (n_x, m)
    assert grads["dh_prev"].shape == (n_h, m)
    assert np.allclose(grad_numeric, grad_analytic, atol=1e-5)
    print('\033[92mTest Aprobado\033[0m')


def gru_backward_test(gru_backward, gru_cell_backward, gru_cell_forward):
    np.random.seed(16)
    n_x, m, T_x, n_h = 2, 2, 3, 2
    x = np.random.randn(n_x, m, T_x)
    h0 = np.random.randn(n_h, m)
    parameters = {
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wh": np.random.randn(n_h, n_h + n_x),
        "bh": np.random.randn(n_h, 1),
        "Wr": np.random.randn(n_h, n_h + n_x),
        "br": np.random.randn(n_h, 1),
    }

    caches_list = []
    h_prev = h0
    for t in range(T_x):
        h_prev, cache = gru_cell_forward(x[:, :, t], h_prev, parameters)
        caches_list.append(cache)
    caches = (caches_list, x)

    dh = np.random.randn(n_h, m, T_x)
    grads = gru_backward(dh, caches, m)

    dx_exp = np.zeros((n_x, m, T_x))
    dh_prevt = np.zeros((n_h, m))
    dWu_exp = np.zeros((n_h, n_h + n_x))
    dWr_exp = np.zeros((n_h, n_h + n_x))
    dWh_exp = np.zeros((n_h, n_h + n_x))
    dbu_exp = np.zeros((n_h, 1))
    dbr_exp = np.zeros((n_h, 1))
    dbh_exp = np.zeros((n_h, 1))
    for t in reversed(range(T_x)):
        g = gru_cell_backward(dh[:, :, t] + dh_prevt, caches_list[t])
        dh_prevt = g["dh_prev"]
        dx_exp[:, :, t] = g["dxt"]
        dWu_exp += g["dWu"]
        dWr_exp += g["dWr"]
        dWh_exp += g["dWh"]
        dbu_exp += g["dbu"]
        dbr_exp += g["dbr"]
        dbh_exp += g["dbh"]
    dWu_exp /= m; dWr_exp /= m; dWh_exp /= m
    dbu_exp /= m; dbr_exp /= m; dbh_exp /= m

    assert np.allclose(grads["dx"], dx_exp)
    assert np.allclose(grads["dWu"], dWu_exp)
    assert np.allclose(grads["dWr"], dWr_exp)
    assert np.allclose(grads["dWh"], dWh_exp)
    assert np.allclose(grads["dbu"], dbu_exp)
    assert np.allclose(grads["dbr"], dbr_exp)
    assert np.allclose(grads["dbh"], dbh_exp)
    print('\033[92mTest Aprobado\033[0m')


def initialize_GRU_parameters_test(initialize_GRU_parameters):
    n_h, n_x, n_y = 3, 2, 1
    params = initialize_GRU_parameters(n_h, n_x, n_y)
    expected_keys = {"Wu", "bu", "Wr", "br", "Wh", "bh", "Wy", "by"}
    assert set(params.keys()) == expected_keys
    assert params["Wu"].shape == (n_h, n_h + n_x)
    assert params["Wy"].shape == (n_y, n_h)
    assert np.all(params["bu"] == 0)
    print('\033[92mTest Aprobado\033[0m')


def model_gru_adam_test(model_gru_adam):
    np.random.seed(17)
    n_x, m, T_x, n_y = 1, 4, 3, 1
    X = np.random.randn(n_x, m, T_x)
    Y = np.random.randn(n_y, m, 1)
    params = model_gru_adam(X, Y, n_h=2, learning_rate=0.01, mini_batch_size=2,
                            num_epochs=2, print_cost=False)
    expected_keys = {"Wu", "bu", "Wr", "br", "Wh", "bh", "Wy", "by"}
    assert set(params.keys()) == expected_keys
    for v in params.values():
        assert np.isfinite(v).all()
    print('\033[92mTest Aprobado\033[0m')


def predict_gru_test(predict_gru):
    np.random.seed(18)
    n_x, m, T_x, n_h, n_y = 1, 2, 3, 2, 1
    x = np.random.randn(n_x, m, T_x)
    parameters = {
        "Wu": np.random.randn(n_h, n_h + n_x),
        "bu": np.random.randn(n_h, 1),
        "Wh": np.random.randn(n_h, n_h + n_x),
        "bh": np.random.randn(n_h, 1),
        "Wr": np.random.randn(n_h, n_h + n_x),
        "br": np.random.randn(n_h, 1),
        "Wy": np.random.randn(n_y, n_h),
        "by": np.random.randn(n_y, 1),
    }

    h_next = np.zeros((n_h, m))
    for t in range(T_x):
        concat = np.concatenate((h_next, x[:, :, t]))
        ut = _sigmoid(np.dot(parameters["Wu"], concat) + parameters["bu"])
        rt = _sigmoid(np.dot(parameters["Wr"], concat) + parameters["br"])
        hht = np.tanh(np.dot(parameters["Wh"], np.concatenate((rt * h_next, x[:, :, t]))) + parameters["bh"])
        h_next = (1 - ut) * h_next + ut * hht
    y_exp = np.dot(parameters['Wy'], h_next) + parameters['by']

    y_pred = predict_gru(x, parameters)
    assert y_pred.shape == (n_y, m)
    assert np.allclose(y_pred, y_exp)
    print('\033[92mTest Aprobado\033[0m')