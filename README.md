# Ground-motion-prediction-using-EHVR
This Python script (GMM.py) processes seismic data, performs predictions of Ground motion, and outputs the results. It uses deep neural network models (ONNX models) in combination with seismic-related parameters and EHVRS for prediction. Here's a detailed guide on how to use this script:

Prerequisites

1.	Install Required Python Libraries: Ensure that the following libraries are installed in your Python environment:

•	pandas for data manipulation.

•	numpy for numerical computations.

•	onnxruntime for loading and executing ONNX models.

•	os for operating system-level interfacing, such as file path operations.

3.	Prepare Data Files:
   
•	You will need a CSV file containing seismic data (refer to temp.csv), the filename of which will be specified by user input.

5.	The model1.csv file needs to be sure it exists and hasn't been changed.
   
Prepare ONNX Model Files:

•	The two ONNX model files should be available, as EHVR_RES_pg_MF2013.onnx, EHVR_RES_sa_MF2013.onnx

Usage Steps

1.	Launch the Script: Run the Python script, GMM.py. It will first prompt you to enter the name of the seismic data file.
2.	Input File Name: Enter the name of the CSV file you wish to process. For example, if the file name is temp.csv, then input that name.

3.	Model Loading and Data Processing: The script will attempt to load the model and process the input CSV data. This includes calculations using the base model and predictions using the ONNX model.
4.	Output Results: Once processing is complete, the script will output two CSV files containing the original data, prediction results, and their combined outcomes. These files will be saved in the output directory specified in the script.
 
Error Handling

•	If there are issues with loading the model or reading the data, the script will prompt error messages. You may need to check the file paths or file formats to ensure they are correct.
Output

•	The script saves the processed results as CSV files, which include all input data, prediction results, and the final combined results. The output files will be located in the designated results directory.
Considerations

•	Ensure that all input paths and filenames are correct.

•	Ensure that the ONNX models used are compatible with the script’s input and output structures.

The temp.csv file contains seismic data, with a focus on different frequency measurements of EHVRs. Here's an explanation of each column and a general idea of the data:

•	site_code: Identifier for the seismic station. 

•	mw: Magnitude of the earthquake.

•	fault_dist: Shortest distance to the fault.

•	eq_location_type_id: A numerical identifier used to categorize the type of earthquake location. 1: CRUSTAL; 2: INTERPLATE; 3: INTRAPLATE. Note that INTRAPLATE is not within the applicable range of this model.

•	Subsequent columns from 0.100Hz to 20.000Hz represent EHVR at specific frequencies measured in Hz. Each of these columns shows the EHVR measurement (non-logarithmic value) at that frequency.


Before using this script, make sure your environment is properly configured and understand how different parts of the code work together so that you can make appropriate adjustments as needed.
