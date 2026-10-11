# Importo las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
import os
import kagglehub
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
import argparse


#Funciones para la importación de la data

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def explore_data(data):
    print(data.head().T)
    print(data.describe())
    print(data.info())

#Funciones para visualización de la información 

def plot_histograms(data):
    data.hist(bins=15, figsize=(15, 10))
    plt.show()

def plot_correlation_matrix(data):
    sns.pairplot(data)
    plt.show()

#Evaluación de sesgo

def skew_evaluation(data, num_cols):
    num_cols_estadistica = {}
    for column in num_cols:
        num_cols_estadistica[column] = data[column].skew()
    
    # Aquí cambiamos num_cols por el diccionario num_cols_estadistica
    skew_df = pd.DataFrame(list(num_cols_estadistica.items()), columns=['Campo', 'Skew']) 

    for i in skew_df.index:    
        if skew_df['Skew'][i] < -1 or skew_df['Skew'][i] > 1:
            skew_df.loc[i, 'Evaluación']  = 'Asimétrica'
        elif (skew_df['Skew'][i] > -1 and skew_df['Skew'][i] < -0.5) or (skew_df['Skew'][i] > 0.5 and skew_df['Skew'][i] < 1):
            skew_df.loc[i, 'Evaluación']  = 'Ligera asimetría'
        else:
            skew_df.loc[i, 'Evaluación']  = 'Aproximadamente simétrica'

    print(skew_df)

#Evaluación de curtósis

def kurt_evaluation(data,cat_cols):

    cat_cols_estadistica = {}

    for column in cat_cols:
        cat_cols_estadistica[column] = data[column].kurt()

    kurt_df = pd.DataFrame(list(cat_cols_estadistica.items()), columns=['Campo', 'Curtosis'])

    for i in kurt_df.index:
        if kurt_df['Curtosis'][i] < -0.5:
            kurt_df.loc[i,'Evaluación'] = 'Platicúrtica'
        elif kurt_df['Curtosis'][i] > 0.5:
            kurt_df.loc[i,'Evaluación'] = 'Leptocúrtica'
        else:
            kurt_df.loc[i,'Evaluación'] ='Aproximadamente mesocúrtica'

    print(kurt_df)

#Generamos gráficas de boxplot
def plot_boxplot(data):
   data.plot(kind='box', subplots=True, layout=(3, 3), figsize=(12, 10), sharex=False, sharey=False)
   plt.tight_layout()
   plt.show()


#Preprocesamiento de los datos: 

# Defino las listas de columnas categóricas y numéricas completas
skewed_cols = ['CCAvg', 'Mortgage', 'Income']
other_num_cols = ['Age', 'Experience']

# Configuro el ColumnTransformer para todas las columnas. 


def scale_and_transform_features(data, target, skewed_cols, other_num_cols):

    # Separar variables dependientes e independientes
    X = data.drop(columns=[target])
    y = data[target].reset_index(drop=True)

    # Seleccionamos el preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            # Numéricas sesgadas -> PowerTransformer
            ('skewed', PowerTransformer(method='yeo-johnson'), skewed_cols),
            
            # Numéricas simétricas -> StandardScaler
            ('scaler', StandardScaler(), other_num_cols)
        ],
        remainder='passthrough'
    )
    
    # Aplicamos la transformación
    features_transformed = preprocessor.fit_transform(X)
    
    # Reconstruimos el DataFrame usando las columnas originales de X
    data_transformed = pd.DataFrame(features_transformed, columns=X.columns)
    
    # Volvemos a pegar la columna target
    data_transformed[target] = y
    
    return data_transformed


#Hacemos la separación de los conjuntos de entrenamiento y test

def split_data(data, target, test_size=0.2, random_state=42):
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


#Hacemos el entrenamiento del modelo

def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

#Hacemos la evaluación del modelo 
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()
    
    report = classification_report(y_test, y_pred)
    print("Classification Report:")
    print(report)

    print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred):.4f}\n")
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred))



#Ahora vamos a la parte de parametrización con argparse


if __name__ == "__main__":
    # PASO 2: Parametrización con argparse
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo de predicción con MLflow')
    parser.add_argument('--C', type=float, default=1.0, help='Inverso de la fuerza de regularización')
    parser.add_argument('--max_iter', type=int, default=1000, help='Número máximo de iteraciones')
    parser.add_argument('--random_state', type=int, default=42, help='Semilla aleatoria')
    parser.add_argument('--filepath', type=str, default='C:\\Users\\marti\\Documents\\MNA\\MLOps\\MLOps actividades\\data\\raw\\Bank_Personal_Loan.csv', help='Ruta del archivo de datos')
    
    args = parser.parse_args()

    # Configuramos el experimento en MLflow (Paso 3 y 4)
    mlflow.set_experiment("experimento_prestamo_logistic")

    with mlflow.start_run():
        # Registrar hiperparámetros con mlflow.log_param
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("filepath", args.filepath)

        # Cargar datos
        df = load_data(args.filepath)

        # Preprocesamiento (usando tus listas de columnas definidas previamente)
        skewed_cols = ['CCAvg', 'Mortgage', 'Income']
        other_num_cols = ['Age', 'Experience']
        target_col = 'Personal_Loan' # Cambia por el nombre real de tu columna objetivo si difiere
        
        df_transformed = scale_and_transform_features(df, target_col, skewed_cols, other_num_cols)

        # Separación de datos usando la semilla parametrizada
        X_train, X_test, y_train, y_test = split_data(df_transformed, target_col, test_size=0.2, random_state=args.random_state)

        # Entrenamiento del modelo con el hiperparámetro C y max_iter
        model = LogisticRegression(C=args.C, max_iter=args.max_iter, random_state=args.random_state)
        model.fit(X_train, y_train)

        # Evaluación y cálculo de métricas
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Registrar métricas en MLflow con mlflow.log_metric
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1_score", f1)

        # Registrar artefactos (ej. guardar y registrar la matriz de confusión como imagen)
        plt.figure(figsize=(6, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Matriz de Confusión")
        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")

        # 8. Registrar el modelo entrenado con MLflow
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        print("Entrenamiento y registro en MLflow completados con éxito")