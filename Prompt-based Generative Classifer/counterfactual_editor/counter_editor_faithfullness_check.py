import os
import pandas as pd

mapping = {
    'Care coordination / communication': 0,
    'Laboratory test': 1,
    'Medication related': 2,
    'Omission / errors in assessment or diagnosis or monitoring': 3,
    'Omission / errors in assessment, diagnosis, monitoring': 3, #comes in two formats
    'Maternal': 4,
    'Equipment / devices': 5,
    'Supplies': 6,
    "<Your single best class here>": 7 #blank prediction
}

results = []
directory = 'output'

for filename in os.listdir(directory): #loop through the output folder to count each file
    if filename.endswith('.csv') and not filename.endswith('_counter.csv'):
        base_path = os.path.join(directory, filename)
        name_no_ext = filename[:-4] #remove the .csv file from the base model's path
        counter_filename = f"{name_no_ext}_counter.csv"
        counter_path = os.path.join(directory, counter_filename)
        
        if not os.path.isfile(counter_path):
            continue
        
        df_base = pd.read_csv(base_path)
        df_counter = pd.read_csv(counter_path)
        df_base_mapped = df_base.replace({'predicted_class': mapping}).reset_index(drop=True)
        df_counter_mapped = df_counter.replace({'predicted_class': mapping}).reset_index(drop=True)
        
        #min_len = min(len(df_base_mapped), len(df_counter_mapped))
        #df_base_mapped = df_base_mapped.iloc[:min_len].reset_index(drop=True)
        #df_counter_mapped = df_counter_mapped.iloc[:min_len].reset_index(drop=True)
        
        '''
        todo: To deal with blank prediction: If both before and after prediction are blank, there is neither information proved that the model's reasoning path remain the same,
        nor the information that the internal reasoning path has changed so we filter it out. But if the prediction has changed, no matter if the blank prediction
        exist or not, it indeed proved the counterfactural has changed the reasoning path 
        '''

        #for now, just remove any null entry
        both_blank_mask = (df_base_mapped['predicted_class'] == 7) & (df_counter_mapped['predicted_class'] == 7) 
        df_base_mapped = df_base_mapped[~both_blank_mask].reset_index(drop=True)
        df_counter_mapped = df_counter_mapped[~both_blank_mask].reset_index(drop=True)

        #count changed predictions
        changed_mask = df_base_mapped['predicted_class'] != df_counter_mapped['predicted_class']
        num_changed = changed_mask.sum()
        results.append([filename, num_changed])

results_df = pd.DataFrame(results, columns=['filename', 'num_changed'])
results_df.to_csv('faithfulness_output_comparison.csv', index=False)
print("Comparison results saved to output_comparison.csv")
