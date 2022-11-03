import mnist.mnist_loader as mnist # Datensatz importieren
import ffnnetwork # Feed Forward Netzwerk importieren (Poster)

training_data, validation_data, test_data = mnist.load_data_wrapper()
training_data = list(training_data)

net = ffnnetwork.Network([784, 30, 10]) # Ein FFN Netzwerk wird erstellt
net.SGD(training_data, 6, 10, 3.0, test_data=test_data) # Das Netwerk fängt an zu lernen