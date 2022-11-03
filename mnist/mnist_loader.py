import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """
    Der Datensatz besteht aus Trainingsdaten und Testdaten.
    Die Trainingsdaten haben insgesamt 50000 Bilder und ein einzelnes
    Bild ist 28*28 Pixel groß. Dies sind eif nur Zahlen von 0-9.
    Die Testdaten haben die gleiche Eigenschaft, nur haben sie nur
    10000 Bilder zum Testen.
    """
    f = gzip.open('mnist/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """
    Wir wandeln die Daten so ab, so dass unser Neuronales Netz damit
    etwas anfangen kann. Unzwar speichern wir dies in einem Tuple,
    mit (x, y). Dabei ist der x der input, also das Bild und der
    y Wert die erwartende Zahl.
    """
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """
    Gibt einen 10 dimensionalen Vektor zurück. Dabei sind alle Werte der
    Dimensionen 0 bis auf die erwartende Zahl.
    Beispiel: Wir wollen, dass die KI die Zahl 3 erratet, dabei muss das
    Ergebnis wie gefolgt aussehen: [0, 0, 0 ,1.0, 0, 0, 0, 0, 0, 0]
    """
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e