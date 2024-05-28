
import pandas as pd
import numpy as np
import os
import onnxruntime as rt

# Define a function to load an ONNX model from a specified file path
def load_onnx_model(onnx_file):
    # Get the current directory path
    current_dir = os.getcwd()
    # Create the full path to the ONNX model file
    model_path = os.path.join(current_dir, onnx_file)
    
    # Check if the model file exists at the path
    if not os.path.exists(model_path):
        # Prompt the user to enter the correct model path if not found
        model_path = input("Please enter the path to the model.onnx: ")
        
        # Raise an error if the model file still can't be found
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
    
    # Return the valid model path
    return model_path

# Prompt the user to input the name of the data file
dt_file = input('Input the file:')

# Read the input data file into a DataFrame
dt = pd.read_csv(dt_file)
# Read another model parameter file into a DataFrame
model1 = pd.read_csv('model1.csv')

# Try to perform calculations on the DataFrame
try:
    # Iterate through each row of the model parameter DataFrame
    for index_shallow, row_model1 in model1.iterrows():
        # Extract parameters needed for the calculation
        pre = row_model1.parameter
        sigma1 = row_model1.sigma1

        # Iterate through each row of the data DataFrame
        for index, row in dt.iterrows():
            # Extract parameters and perform calculations based on model formulas
            a1 = row_model1['a1']
            Mw1_min = min(row['mw'], row_model1['Mw01']) 
            Mw1 = row_model1['Mw1']
            eq_type = row['eq_location_type_id']
            b1_k = row_model1[f'b1_{eq_type}']
            c1_k = row_model1[f'c1_{eq_type}']
            X = row['fault_dist']
            d1 = row_model1['d1']
            e1 = row_model1['e1']    

            # Store calculated result back into the DataFrame
            dt.loc[index, f'Basemodel_{pre}'] = a1 * ((Mw1_min - Mw1) ** 2) + b1_k * X + c1_k - np.log10(X + d1 * (10 ** (e1 * Mw1_min)))

# Catch any exceptions that occur during the process
except Exception as e:
    print("Error reading specific columns:", e)

# Ensure the output directory exists
output_folder = os.path.join(os.getcwd(), 'result')
os.makedirs(output_folder, exist_ok=True)
# Define the filename for the ONNX model
onnx_file_pg = ('EHVR_RES_pg_MF2013.onnx')

# Define the path to the input CSV file
input_folder = ("MTE Database/GMM/all/pEHVR_test_pg.csv")

# Load the ONNX model
model_file = load_onnx_model(onnx_file_pg)
sess = rt.InferenceSession(model_file)
# Retrieve the model's input and output node names
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Prepare the test data from the DataFrame
X_test = dt[['Basemodel_PGA','Basemodel_PGV',
      '0.100Hz', '0.105Hz', '0.111Hz', '0.118Hz', '0.125Hz', '0.133Hz', 
       '0.143Hz', '0.154Hz', '0.167Hz', '0.182Hz', '0.200Hz', '0.222Hz',
       '0.250Hz', '0.286Hz', '0.333Hz', '0.400Hz', '0.455Hz', '0.500Hz',
       '0.588Hz', '0.667Hz', '0.769Hz', '0.833Hz', '0.909Hz', '1.000Hz',
       '1.111Hz', '1.250Hz', '1.429Hz', '1.667Hz', '2.000Hz', '2.222Hz',
       '2.500Hz', '2.857Hz', '3.333Hz', '4.000Hz', '4.545Hz', '5.000Hz',
       '6.667Hz', '7.692Hz', '8.333Hz', '9.091Hz', '10.000Hz', '11.111Hz',
       '12.500Hz', '14.286Hz', '16.667Hz', '20.000Hz']].values

# Create an empty DataFrame to store prediction results
Y_result_PG = pd.DataFrame()

# Iterate through each row of test data and make predictions using the ONNX model
for row in X_test:
    # Run prediction and reshape the data as required by the model
    pred_onx = sess.run([output_name], {input_name: row.reshape(1, 48).astype(np.float32)})[0]
    # Append predictions to the result DataFrame
    Y_result_PG = pd.concat([Y_result_PG, pd.DataFrame(pred_onx)], ignore_index=True)

# Set the column names for the prediction results
Y_result_PG.columns = ['pre_DNN_PGA', 'pre_DNN_PGV']

# Re-read the complete CSV file to ensure all data is available
X_temp = dt

# Merge the prediction results with the original data
result = pd.concat([X_temp, Y_result_PG], axis=1)

# Specify another model file
onnx_file_sa = ('EHVR_RES_sa_MF2013.onnx')

# Load the second ONNX model
model_file = load_onnx_model(onnx_file_sa)
sess = rt.InferenceSession(model_file)
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name

# Define the test data file
test_file = input_folder
# Read the CSV file and prepare the test data
X_test_sa = dt[['Basemodel_0.05', 'Basemodel_0.06', 'Basemodel_0.07', 'Basemodel_0.08', 'Basemodel_0.09', 'Basemodel_0.1', 
                'Basemodel_0.11', 'Basemodel_0.12', 'Basemodel_0.13', 'Basemodel_0.15',  'Basemodel_0.2', 'Basemodel_0.22', 
                'Basemodel_0.25', 'Basemodel_0.3', 'Basemodel_0.35', 'Basemodel_0.4', 'Basemodel_0.45', 'Basemodel_0.5', 
                'Basemodel_0.6', 'Basemodel_0.7', 'Basemodel_0.8', 'Basemodel_0.9', 'Basemodel_1', 'Basemodel_1.1', 
                'Basemodel_1.2', 'Basemodel_1.3', 'Basemodel_1.5', 'Basemodel_1.7', 'Basemodel_2', 'Basemodel_2.2', 
                'Basemodel_2.5', 'Basemodel_3', 'Basemodel_3.5', 'Basemodel_4', 'Basemodel_4.5', 'Basemodel_5',
                'Basemodel_5.5', 'Basemodel_6', 'Basemodel_6.5', 'Basemodel_7', 'Basemodel_7.5', 'Basemodel_8', 
                'Basemodel_8.5', 'Basemodel_9', 'Basemodel_9.5', 'Basemodel_10',

                  '0.100Hz', '0.105Hz', '0.111Hz', '0.118Hz', '0.125Hz', '0.133Hz', 
                   '0.143Hz', '0.154Hz', '0.167Hz', '0.182Hz', '0.200Hz', '0.222Hz',
                   '0.250Hz', '0.286Hz', '0.333Hz', '0.400Hz', '0.455Hz', '0.500Hz',
                   '0.588Hz', '0.667Hz', '0.769Hz', '0.833Hz', '0.909Hz', '1.000Hz',
                   '1.111Hz', '1.250Hz', '1.429Hz', '1.667Hz', '2.000Hz', '2.222Hz',
                   '2.500Hz', '2.857Hz', '3.333Hz', '4.000Hz', '4.545Hz', '5.000Hz',
                   '6.667Hz', '7.692Hz', '8.333Hz', '9.091Hz', '10.000Hz', '11.111Hz',
                   '12.500Hz', '14.286Hz', '16.667Hz', '20.000Hz']].values

# Create an empty DataFrame to store the prediction results for spectral accelerations
Y_result_sa = pd.DataFrame()

# Iterate through each row of test data for spectral accelerations and make predictions using the ONNX model
for row in X_test_sa:
    # Run prediction and reshape the data as required by the model
    pred_onx = sess.run([output_name], {input_name: row.reshape(1, 92).astype(np.float32)})[0]
    # Append predictions to the result DataFrame
    Y_result_sa = pd.concat([Y_result_sa, pd.DataFrame(pred_onx)], ignore_index=True)

# Set the column names for the prediction results for spectral accelerations
Y_result_sa.columns = ['pre_DNN_0.05', 'pre_DNN_0.06', 'pre_DNN_0.07', 'pre_DNN_0.08', 'pre_DNN_0.09', 'pre_DNN_0.1', 
                        'pre_DNN_0.11', 'pre_DNN_0.12', 'pre_DNN_0.13', 'pre_DNN_0.15',  'pre_DNN_0.2', 'pre_DNN_0.22', 
                        'pre_DNN_0.25', 'pre_DNN_0.3', 'pre_DNN_0.35', 'pre_DNN_0.4', 'pre_DNN_0.45', 'pre_DNN_0.5', 
                        'pre_DNN_0.6', 'pre_DNN_0.7', 'pre_DNN_0.8', 'pre_DNN_0.9', 'pre_DNN_1', 'pre_DNN_1.1', 
                        'pre_DNN_1.2', 'pre_DNN_1.3', 'pre_DNN_1.5', 'pre_DNN_1.7', 'pre_DNN_2', 'pre_DNN_2.2', 
                        'pre_DNN_2.5', 'pre_DNN_3', 'pre_DNN_3.5', 'pre_DNN_4', 'pre_DNN_4.5', 'pre_DNN_5',
                        'pre_DNN_5.5', 'pre_DNN_6', 'pre_DNN_6.5', 'pre_DNN_7', 'pre_DNN_7.5', 'pre_DNN_8', 
                        'pre_DNN_8.5', 'pre_DNN_9', 'pre_DNN_9.5', 'pre_DNN_10'  ]

# Merge the prediction results with the original data to get the final result
result = pd.concat([result, Y_result_sa], axis=1)

# Prepare a list of frequencies to be used in final calculations
fre_list = ['PGA', 'PGV',
            '0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '0.11', '0.12', '0.13', '0.15', 
            '0.2', '0.22', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.6',          
            '0.7', '0.8', '0.9', '1', '1.1', '1.2', '1.3', '1.5', '1.7', '2', '2.2',          
            '2.5', '3', '3.5', '4', '4.5', '5',         
            '5.5', '6', '6.5', '7', '7.5', '8', '8.5', '9', '9.5', '10']

# Perform the final calculation by adding the base model result to the prediction result
for fre in fre_list:
    result[f'Final_{fre}'] = result[f'Basemodel_{fre}'] + result[f'pre_DNN_{fre}']

# Generate the output file name from the original data file name
base_name = os.path.basename(dt_file)
output_file_name = f"result_{base_name}"
# Save the final result DataFrame to a CSV file in the specified output folder
result.to_csv(os.path.join(output_folder, output_file_name), index=False, float_format='%.6f')

# Define the frequencies for the data transformation
frequencies = [
    'PGA', 'PGV', '0.05', '0.06', '0.07', '0.08', '0.09', '0.1', '0.11', '0.12', '0.13', '0.15',
    '0.2', '0.22', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.6',
    '0.7', '0.8', '0.9', '1', '1.1', '1.2', '1.3', '1.5', '1.7', '2', '2.2',
    '2.5', '3', '3.5', '4', '4.5', '5',
    '5.5', '6', '6.5', '7', '7.5', '8', '8.5', '9', '9.5', '10'
]

# Prepare the transformed data
output_data = []
for index, row in result.iterrows():
    base_info = row[['site_code', 'mw', 'fault_dist', 'eq_location_type_id']]
    for freq in frequencies:
        new_row = {
            'site_code': base_info['site_code'],
            'mw': base_info['mw'],
            'fault_dist': base_info['fault_dist'],
            'eq_location_type_id': base_info['eq_location_type_id'],
            'fre': freq,
            'Basemodel': row[f'Basemodel_{freq}'],
            'pre_DNN': row[f'pre_DNN_{freq}'],
            'Final': row[f'Final_{freq}']
        }
        output_data.append(new_row)

# Convert to DataFrame
output_df = pd.DataFrame(output_data)
output_df_name = f"result_all_{base_name}"
output_df.to_csv(os.path.join(output_folder, output_df_name), index=False, float_format='%.6f')

print(f"Results have been saved as {output_folder}/{output_file_name}")
print(f"Results have been saved as {output_folder}/{output_df_name}")
input()
