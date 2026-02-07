import openml
import pandas as pd
import os
import requests

# Configuration
SAVE_DIR = "./mini_test_datasets"
OPENML_SUBDIR = os.path.join(SAVE_DIR, "openml")

# Ensure directories exist
os.makedirs(OPENML_SUBDIR, exist_ok=True)

# PART 1: Download OpenML Datasets

def download_openml_data():
    target_datasets = [
        "pendigits", 
        "credit-approval", 
        "banknote-authentication", 
        "diabetes"
    ]
    
    print(f"--- Starting OpenML Downloads ({len(target_datasets)} datasets) ---")
    
    # Connect to OpenML-CC18 suite (ID 99)
    try:
        suite = openml.study.get_suite(suite_id=99)
    except Exception as e:
        print(f"Error connecting to OpenML: {e}")
        return

    for task_id in suite.tasks:
        try:
            task = openml.tasks.get_task(task_id, download_data=False)
            dataset = task.get_dataset()
            
            if dataset.name in target_datasets:
                print(f"Downloading {dataset.name}...")
                
                # Fetch data
                X, y, categorical_indicator, attribute_names = dataset.get_data(
                    target=dataset.default_target_attribute,
                    dataset_format='dataframe'
                )
                
                # Combine features and target into one DataFrame
                full_df = pd.concat([X, y], axis=1)
                
                # Save to CSV
                save_path = os.path.join(OPENML_SUBDIR, f"{dataset.name}.csv")
                full_df.to_csv(save_path, index=False)
                print(f"   -> Saved to {save_path}")
                
        except Exception as e:
            print(f"   ! Failed to download task {task_id}: {e}")


# Main Execution
if __name__ == "__main__":
    download_openml_data()
    print("\nAll downloads complete.")