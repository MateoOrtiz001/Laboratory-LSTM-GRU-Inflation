import numpy as np
import random


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