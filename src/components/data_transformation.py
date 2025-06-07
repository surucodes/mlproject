# Import necessary libraries for data transformation, error handling, and logging
import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # Used to apply different transformations to different columns
from sklearn.impute import SimpleImputer  # Used to fill missing values
from sklearn.pipeline import Pipeline  # Used to chain preprocessing steps
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For encoding categorical data and scaling numerical data
from src.utils import save_object  # Custom utility to save objects (e.g., preprocessor) as .pkl files
from src.exception import CustomException  # Custom exception for detailed error reporting
from src.logger import logging  # Custom logging for tracking progress and errors

@dataclass
class DataTransformationConfig:
    # Define the file path where the preprocessor object will be saved
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")
    # Why: Centralizes configuration settings (e.g., file paths) for easy access and modification

class DataTransformation:
    def __init__(self):
        # Initialize the class with a DataTransformationConfig object
        self.data_transformation_config = DataTransformationConfig()
        # Why: Stores config settings (e.g., preprocessor file path) for use in transformation methods

    def get_data_transformer_object(self):
        '''
        This function creates a preprocessing object for numerical and categorical data.
        It defines pipelines for imputing missing values and transforming features.
        '''
        try:
            # Define numerical and categorical columns based on the dataset
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            # Why: Different columns need different transformations (numerical: scale, categorical: encode)

            # Pipeline for numerical columns: impute missing values and scale
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with column median
                    ("scaler", StandardScaler(with_mean = False))  # Standardize to mean=0, std=1
                ]
            )
            # Why: Ensures numerical features are complete and on the same scale for ML models

            # Pipeline for categorical columns: impute missing values and one-hot encode
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with most common category
                    ("one_hot_encoder", OneHotEncoder())  # Convert categories to binary columns (e.g., male=1, female=0)
                    # Note: Removed StandardScaler because one-hot encoded data is already binary (0/1)
                ]
            )
            # Why: Converts categorical data to numerical format suitable for ML models

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)  # Fixed naming from cat_pipelines to cat_pipeline
                ]
            )
            # Why: Applies the correct pipeline to the corresponding columns in one step

            # Log the creation of the preprocessor
            logging.info("Preprocessing object created successfully")
            return preprocessor
            # Why: Returns the preprocessor to be used for transforming data
            
        except Exception as e:
            # Catch any errors and raise a CustomException with detailed info
            raise CustomException(e, sys)
            # Why: Integrates with your error handling system for debugging

    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function reads train/test data, applies transformations, and returns transformed arrays.
        It also saves the preprocessor object for future use (e.g., in prediction).
        '''
        try:
            # Read training and test CSV files into pandas dataframes
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            # Why: Loads the data prepared by data_ingestion.py

            # Get the preprocessing object
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()
            # Why: Prepares the transformer to apply consistent transformations

            # Define the target column and numerical columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]
            # Why: Identifies the target (to predict) and numerical features

            # Split training data into features (X) and target (y)
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            # Why: Separates features for transformation and target for model training

            # Split test data into features (X) and target (y)
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            # Why: Ensures test data is prepared the same way for evaluation

            # Apply preprocessing to training and test features
            logging.info("Applying preprocessing object on training and testing dataframes")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            # This applies all the preprocessing steps (imputing missing values, one-hot encoding, scaling, etc.) to your training data,
            # and then applies the same fitted method to the test data.
            # Why: fit_transform learns parameters (e.g., median, scaling factors) on training data; transform applies them to test data

            # Combine transformed features with target variable into arrays
            logging.info("Combining transformed features with target")
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            # np.c_ concatenates the transformed features (arrays) with the target variable (converted to array).
            # Why: Creates arrays with preprocessed, transformed features and target together, ready for model training.

            # Save the preprocessing object for future use
            logging.info("Saving preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            # We save this so we can use the same preprocessing object in predict.py or when new data comes in,
            # applying the same preprocessing should be done.

            logging.info("Data transformation completed")
            # Why: Logs completion for tracking

            # Return transformed arrays and preprocessor file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            # These are used by model_trainer.py for training and by other pipelines for prediction.

        except Exception as e:
            # Catch any errors and raise a CustomException
            raise CustomException(e, sys)
            # Why: Ensures errors are reported with detailed messages for debugging







# #any code related to transformation of data eg : one hpt encoding , numerical to categorical etc
# import sys 
# from dataclasses import dataclass
# import os
# import numpy as np
# import pandas as pd
# from sklearn.compose import ColumnTransformer
# #is used to do pipeline 
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import OneHotEncoder,StandardScaler
# from src.utils import save_object
# from src.exception import CustomException
# from src.logger import logging

# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config  = DataTransformationConfig()

#     def get_data_transformer_object(self):
#         '''
#         This function is for data transformation for numerical and categorical data.
        
        
#         '''
#         try:
#             numerical_columns = ["writing_score","reading_score"]
#             categorical_columns = [
#                 "gender",
#                 "race_ethnicity",
#                 "parental_level_of_education",
#                 "lunch",
#                 "test_preparation_course",
#             ]
#             num_pipeline = Pipeline(
#                 steps = [
#                     ("imputer",SimpleImputer(strategy = "median")),
#                     ("scaler",StandardScaler())

#                 ]
#             )
#             cat_pipeline = Pipeline(
#                 steps = [
#                     ("imputer",SimpleImputer(strategy="most_frequent")),
#                     ("one_hot_encoder",OneHotEncoder()),
#                     ("scaler",StandardScaler())
#                 ]
#             )
#             logging.info("Numerical columns standard scaling completed")
#             logging.info("Categorical columns encoding completed")

#             preprocessor = ColumnTransformer(
#                 [

#                 ("num_pipeline",num_pipeline,numerical_columns),
#                 ("cat_pipelines",cat_pipeline,categorical_columns)

#                 ]


#             )
#             return preprocessor
            
#         except Exception as e:
#             raise CustomException(e , sys)
        
#     def initiate_data_transformation(self,train_path ,test_path):

#         try :
#             train_df = pd.read_csv(train_path)
#             test_df = pd.read_csv(test_path)

#             logging.info("Read train and test data completed")
#             logging.info("Obtaining preprocessing object")

#             preprocessing_obj = self.get_data_transformer_object()

#             target_column_name = "math_score"
#             numerical_columns = ["writing_score","reading_score"]

#             input_feature_train_df = train_df.drop(columns= [target_column_name],axis=1)
#             target_feature_train_df = train_df[target_column_name]

#             input_feature_test_df = test_df.drop(columns=[target_column_name],axis = 1)
#             target_feature_test_df = test_df[target_column_name]

#             logging.info(
#                 f"Applying preprocessing object on training dataframe and testing dataframe."
#             )

#             input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
#             input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
#             # This applies all the preprocessing steps (imputing missing values, one hot encoding, scaling, etc...) to your training data, and then applied the same fitted method to the test data.
#             train_arr = np.c [ input_feature_train_arr , np.array(target_feature_train_df)]
#             test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df) ]
#             #np.c_ concatenates the transformed features (arrays) with the target variable (converted to array).
#             # Why: Creates arrays with preprocessed, transformed features and target together, ready for model training.
#             logging.info(f"Saved preprocessing object.")
#             # we save this so we can use the same preprocessing object in predict.py or when new data comes in , applying the same preprocessing should be done.
#             save_object(
#                 file_path = self.data_transformation_config.preprocessor_obj_file_path,
#                 obj = preprocessing_obj

#             )
#             return (
#                     train_arr,
#                     test_arr , 
#                     self.data_transformation_config.preprocessor_obj_file_path,
#         # : These are used by model_trainer.py for training and by other pipelines for prediction.

#             )
#         except Exception as e :
#             raise CustomException(e,sys) 
        


