# Try various number of hidden nodes and see what performs best
for i in range(5, 50, 5):
    nn = OCRNeuralNetwork(i, data_matrix, data_labels, train_indices, False)
    performance = str(test(data_matrix, data_labels, test_indices, nn))
    print('{i}Hidden Nodes: {val}'.format(i=i, val=performance))


def test(data_matrix, data_labels, test_indices, nn):
    avg_sum = 0
    
    for j in range(100):
        correct_guess_count = 0
        
        for i in test_indices:
            test = data_matrix[i]
            prediction = nn.predict(test)
            
            if data_labels[i] == prediction:
                correct_guess_count += 1
    
        avg_sum += (correct_guess_count / float(len(test_indices)))
    
    return avg_sum / 100