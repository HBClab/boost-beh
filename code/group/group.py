import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Group:

    def __init__(self) -> None:
        pass

    @staticmethod
    def load_task_data(task_name, relevant_columns, root_dir="../../../data"):
        """
        Loads all subject CSVs corresponding to a specified task, extracts relevant columns, 
        and appends them (with the subject ID as the first column) into one DataFrame.
        
        Parameters:
            task_name (str): The name of the task folder (e.g., "AF", "NTS", "DWL").
            relevant_columns (list): A list of column names (strings) to extract from each CSV.
            root_dir (str): The root data directory. Default is "Data".
            
        Returns:
            pandas.DataFrame: A DataFrame containing the subject ID (as 'subjectID') and the relevant columns 
                            from every CSV that was found.
        """
        # List to collect dataframes for each subject
        data_frames = []
        
        # Loop through each study directory in the root directory
        for study in os.listdir(root_dir):
            study_path = os.path.join(root_dir, study)
            if os.path.isdir(study_path):
                # Loop through each site directory within the study directory
                for site in os.listdir(study_path):
                    site_path = os.path.join(study_path, site)
                    if os.path.isdir(site_path):
                        # Loop through each subject directory within the site directory
                        for subject in os.listdir(site_path):
                            subject_path = os.path.join(site_path, subject)
                            if os.path.isdir(subject_path):
                                # Construct the path to the task directory for this subject
                                task_path = os.path.join(subject_path, task_name)
                                if os.path.isdir(task_path):
                                    # Look for the 'data' folder within the task directory
                                    data_folder = os.path.join(task_path, "data")
                                    if os.path.isdir(data_folder):
                                        # Process each CSV file found in the data folder
                                        for file in os.listdir(data_folder):
                                            # if file ends with .csv and second part of the filename is 'ses-1'
                                            if file.endswith("2.csv"):
                                                continue
                                            elif file.endswith(".csv") and "_cat-1_" in file:
                                                csv_path = os.path.join(data_folder, file)
                                                print(f"Processing {csv_path}")
                                            # if the character before .csv is 2 then skip
                                                try:
                                                    # Load the CSV into a temporary DataFrame
                                                    temp_df = pd.read_csv(csv_path)
                                                    # Select only the relevant columns (if they exist)
                                                    # It is assumed that every CSV contains all desired columns; 
                                                    # you may wish to add error handling if some files do not.
                                                    filtered_df = temp_df[relevant_columns].copy()
                                                    # Insert the subject ID as the first column
                                                    filtered_df.insert(0, "subjectID", subject)
                                                    # Append the DataFrame for this subject to the list
                                                    data_frames.append(filtered_df)
                                                except Exception as e:
                                                    print(f"Error processing {csv_path}: {e}")
                                                finally:
                                                    # Delete the temporary dataframe to free memory
                                                    del temp_df
                                                    del filtered_df
                                            
        # Concatenate all subject DataFrames into one, resetting the index
        if data_frames:
            final_df = pd.concat(data_frames, ignore_index=True)
        else:
            # If no data was found, return an empty DataFrame with the appropriate columns
            final_df = pd.DataFrame(columns=["subjectID"] + relevant_columns)
        
        return final_df

    def return_dfs(self):
        pass

