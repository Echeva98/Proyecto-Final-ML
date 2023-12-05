from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    accuracy = calculate_accuracy(y_test, predictions)

    return accuracy

from model_evaluation import evaluate_model

model1 = load_model('modelo1.h5')
accuracy1 = evaluate_model(model1, X_test_scaled, y_test)

model2 = load_model('modelo2.h5')
accuracy2 = evaluate_model(model2, X_test, y_test)
... 



