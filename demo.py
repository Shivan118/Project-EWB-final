from Insurance.pipeline.batch_prediction import start_batch_prediction
from Insurance.pipeline.training_pipeline import start_training_pipeline

file_path=r"C:\Users\shiva\Desktop\Project Template ML\Day_8\Project-EWB_model_batch_prediction\insurance.csv"
print(__name__)
if __name__=="__main__":
    try:
        #output_file = start_training_pipeline()
        output_file = start_batch_prediction(input_file_path=file_path)
        print(output_file)
    except Exception as e:
        print(e)