import os
from typing import Optional
from Insurance.entity.config_entity import TRANSFORMER_OBJECT_FILE_NAME,MODEL_FILE_NAME,TARGET_ENCODER_OBJECT_FILE_NAME

# Folder ( new model saved )
# comparision new model vs old model
# accpedted else rejected


# save_model -> 0 -> 1 -> 2 -> 3

class ModelResolver:
    def __init__(self, model_registry:str = "saved_models", #
                transfomer_dir_name = "transfomer",
                traget_encoder_dir_name = "target_encoder",
                model_dir_name = "model"):

        self.model_registry = model_registry
        os.makedirs(self.model_registry, exist_ok=True)
        self.transfomer_dir_name = transfomer_dir_name
        self.traget_encoder_dir_name = traget_encoder_dir_name
        self.model_dir_name = model_dir_name

# 1
    def get_latest_dir_path(self)->Optional[str]:
        try:
            dir_name = os.listdir(self.model_registry)
            if len(dir_name) == 0:
                return None
            
            dir_name = list(map(int, dir_name))
            latest_dir_name = max(dir_name)
            return os.path.join(self.model_registry, f"{latest_dir_name}")

        except Exception as e:
            raise e
# 2

    def get_latest_model_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Model is Not avaliable")

            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME)
        except Exception as e:
            raise e
# 3

    def get_latest_transformer_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Transform data is not avaliable")

            # return os.path.join(latest_dir, self.traget_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
            return os.path.join(latest_dir, self.transformer_dir_name,TRANSFORMER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e

# 4
    def get_latest_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir is None:
                raise Exception(f"Traget encoder data is not avaliable")

            return os.path.join(latest_dir, self.traget_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME)
        except Exception as e:
            raise e
# 5
    def get_latest_save_dir_path(self)->str:

        try:
            latest_dir = self.get_latest_dir_path()
            if latest_dir ==  None: #Here replace is to ==
                return os.path.join(self.model_registry, f"{0}")

            latest_dir_num = int(os.path.basename(self.get_latest_dir_path()))
            return os.path.join(self.model_registry, f"{latest_dir_num + 1}") ## add 1 so that it will increase eveytime
        except Exception as e:
            raise e
# 6
    def get_latest_save_model_path(self):

        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.model_dir_name, MODEL_FILE_NAME) # model.pkl
        except Exception as e:
            raise e
# 7
    def get_latest_save_transfomer_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.transfomer_dir_name, TRANSFORMER_OBJECT_FILE_NAME) # transform.pkl
        except Exception as e:
            raise e

# 8
    def get_latest_save_target_encoder_path(self):
        try:
            latest_dir = self.get_latest_save_dir_path()
            return os.path.join(latest_dir, self.traget_encoder_dir_name, TARGET_ENCODER_OBJECT_FILE_NAME) # encoder.pkl
        except Exception as e:
            raise e

