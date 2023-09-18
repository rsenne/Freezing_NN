import torch
import pandas as pd
import numpy as np
from FNN import freezingmodel

all = ["dataProcessing"]


class dataProcessing:
    def __init__(self, end_time, df):
        self.df = df
        self.end_time = end_time
        self.timestamps = np.linspace(0, self.end_time, len(self.df))
        self.predictions = None
        self.pseudo_file = None
    
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = freezingmodel.net(174).to(self.device)
        
        self.model.load_state_dict(torch.load("freeze_network_model_params.pth"))

        if torch.cuda.is_available():
            print("Succesfully using GPU.")
        else:
            print("Using CPU, consider restarting kernel.")

    def seconds_to_timestamp(self, seconds):
        """Converts a floating point seconds value into the desired timestamp format."""
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return "{:02d}:{:02d}:{:06.4f}".format(int(hours), int(minutes), seconds)

    def evaluate(self):
        self.model.eval()
        tensor = torch.tensor(self.df.to_numpy(), dtype=torch.float32).to(self.device)
        # Disables gradient calculation to save memory
        with torch.no_grad():
            self.predictions = self.model(tensor).squeeze().round().cpu().numpy()

    def pseudo_anymaze(self):
        # Convert the data into a dataframe
        df = pd.DataFrame({
        'Freezing': self.predictions.astype(int),  # Convert to integer
        'Time': [self.seconds_to_timestamp(t) for t in self.timestamps]
        })

        # Calculate the difference between rows for the 'Freezing' column
        df['Diff'] = df['Freezing'].diff().fillna(df['Freezing'].iloc[0])

        # Filter rows where the difference is not zero (i.e., transition points)
        transitions_df = df[df['Diff'] != 0].reset_index(drop=True)

        # Prepare final dataframe
        final_data = []

        for i in range(0, len(transitions_df)-1, 2):
            start_time = transitions_df['Time'].iloc[i]
            freezing = transitions_df['Freezing'].iloc[i]

            if i+1 < len(transitions_df):  # check if there's a next transition
                end_time = transitions_df['Time'].iloc[i+1]

                # Convert timestamps to seconds to calculate difference
                start_seconds = sum(float(x) * 60 ** idx for idx, x in enumerate(reversed(start_time.split(":"))))
                end_seconds = sum(float(x) * 60 ** idx for idx, x in enumerate(reversed(end_time.split(":"))))
            
                # Check if bout length is at least 2.5 seconds
                if end_seconds - start_seconds >= 2.5:
                    final_data.append([freezing, start_time])
                    final_data.append([0, end_time])

        result = pd.DataFrame(final_data, columns=['Freezing', 'Time'])
        self.pseudo_file = result
        
