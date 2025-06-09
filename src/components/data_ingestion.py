#this ingests the dataset from whereever it might ve
#also train test validation split is done in here
import os 
import sys
from src.exception import CustomException
from src.logger import logging 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.utils import save_object
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts',"train.csv")
    test_data_path: str = os.path.join('artifacts',"test.csv")
    raw_data_path: str = os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method/component")
        try:
            df= pd.read_csv('notebook/StudentsPerformance.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header= True )
            
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header= True )

            logging.info("Ingestion of the data is completed")

            return (

                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                #for data transformation
            )
        except Exception as e:
            raise CustomException(e,sys)
            
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    #This calls the constructor which creates a file path for the pickle file that is going to store the preprocessing object that is the column transformer.
    train_array,test_array,_ = data_transformation.initiate_data_transformation(train_data,test_data)
# the initiate_data_transformation method , in the data_transformation object, of the DataTransformation class, retrieves the train and test data as an output of the data ingestion module and then applies the column transformer (which in itself has pipelines for the num and cat data) to the target variable separated train and test dataframe and then concatenates them back using np.c_ and then returns the trasformed train and test array with the preprocessing object saved finally inorder to apply it to the new data

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_array,test_array))
    