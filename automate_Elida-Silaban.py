import pandas as pd

def preprocess_data(input_file, output_file):
    data = pd.read_csv(input_file, sep=';')
    data = data.drop_duplicates()
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    preprocess_data(
        "/content/winequality-red.csv",
        "wine_preprocessing.csv"
    )
