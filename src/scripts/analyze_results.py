import os
from glob import glob
import pandas as pd

def parse_filename(filename):
    """
    Parses the filename to extract model parameters.
    
    Parameters:
    - filename (str): The path to the filename from which to extract parameters.
    
    Returns:
    - dict: A dictionary containing parsed parameters such as model name, epochs, number of convolutional layers,
            optimizer, learning rate, step size, weight decay, training accuracy, and test accuracy.
            Returns None if the model name does not match expected values.
    """
    base = os.path.basename(filename)
    parts = base.split('_')
    model = parts[1]
    if "FlexibleConvLayer" in model or "FlexibleResNet" in model:
        parameters = {
            'model': model,
            'epochs': parts[3],
            'num_conv_layers': parts[5],
            'optimizer': parts[7],
            'lr': parts[9],
            'step': parts[11],
            'weight_decay': parts[13],
            'train_acc': parts[15],
            'test_acc': float(parts[17].split('.')[0] + '.' + parts[17].split('.')[1][:2])
        }
        return parameters
    return None

def load_data(result_path):
    """
    Loads data from files in a specified directory, extracting information based on file naming conventions.
    
    Parameters:
    - result_path (str): Directory path where result files are stored.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the loaded and parsed data.
    """
    filenames = glob(os.path.join(result_path, "*.png"))
    data = []
    for filename in filenames:
        parameters = parse_filename(filename)
        if parameters:
            data.append(list(parameters.values()))
    columns = ['Model', 'Epochs', 'NumConvLayers', 'Optimizer', 'LR', 'Step', 'WeightDecay', 'TrainAcc', 'TestAcc']
    df = pd.DataFrame(data, columns=columns)
    return df

def save_sorted_df(df, result_path, filename='sorted_results_all.csv'):
    """
    Sorts a DataFrame based on specified columns and saves it to a CSV file.
    
    Parameters:
    - df (DataFrame): The DataFrame to sort and save.
    - result_path (str): Directory path to save the sorted DataFrame.
    - filename (str): Filename for the saved CSV file.
    """
    sorted_df = df.sort_values(by=['Model', 'Epochs', 'NumConvLayers', 'Optimizer', 'LR', 'Step', 'WeightDecay', 'TrainAcc', 'TestAcc'])
    csv_file_path = os.path.join(result_path, filename)
    sorted_df.to_csv(csv_file_path, index=False)
    print(f"DataFrame sorted by parameters saved as CSV in {csv_file_path}")

def save_top_performances(df, result_path, top_n=10):
    """
    Identifies and saves the top N model performances to CSV files, one for each model type.
    
    Parameters:
    - df (DataFrame): The DataFrame containing model performance data.
    - result_path (str): Directory path to save the top performance CSV files.
    - top_n (int): Number of top performances to save for each model type.
    """
    model_types = df['Model'].unique()
    for model_type in model_types:
        df_filtered = df[df['Model'].str.contains(model_type)].sort_values(by='TestAcc', ascending=False).head(top_n)
        csv_file_path = os.path.join(result_path, f'top{top_n}_{model_type}_accuracy.csv')
        df_filtered.to_csv(csv_file_path, index=False)
        print(f"Top {top_n} accuracy results for {model_type} saved as CSV in {csv_file_path}")

def save_global_top10(df, result_path, filename='global_top10_accuracy_results.csv'):
    """
    Saves the global top 10 performances across all models based on test accuracy.
    
    Parameters:
    - df (DataFrame): The DataFrame containing model performance data.
    - result_path (str): Directory path to save the global top 10 performance CSV file.
    - filename (str): Filename for the saved CSV file.
    """
    df_top10_global = df.sort_values(by='TestAcc', ascending=False).head(10)
    csv_file_path = os.path.join(result_path, filename)
    df_top10_global.to_csv(csv_file_path, index=False)
    print(f"Global top 10 accuracy results saved as CSV in {csv_file_path}")

def main():
    """
    Main function to execute the script functionality: load data, sort and save the DataFrame,
    save top model performances, and save global top 10 performances.
    """
    result_path = "./results/graphs"
    save_path = "./results"
    
    df = load_data(result_path)
    save_sorted_df(df, save_path)
    save_top_performances(df, save_path)
    save_global_top10(df, save_path)

if __name__ == "__main__":
    main()
