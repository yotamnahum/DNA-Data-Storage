import json 
import datetime, os
import sys

config_path = "srrtransformer/config.json"

def make_config(config=None):
    """
    *Creates the config.json file that holds all configuration relevant for model* 
    
        parameters:
        ----------
            :param configs:
                type: dictionary
                Default: None
                A dictionary with all config information. If none, sets default config parameters.

        Attributes:
        --------
        Creates the config.json file
        """
    if config is None:
        config = {
            "PROJECT_FOLDER": ".",
            "data_type": "text",
            "k_mer_size": 4,
            "seq_length": 100,
            "pretrained_model_path": None,
            "save_loacally": True,
            "random_state": 42,
            "extrnal_model":None,
            "unique_experiment_path":None,
            }
        
    PROJECT_FOLDER = config["PROJECT_FOLDER"]
    data_type = config["data_type"] 
    k_mer_size = config["k_mer_size"] 
    seq_length = config["seq_length"] 
    pretrained_model_path = config["pretrained_model_path"] 
    save_loacally = config["save_loacally"] 
    random_state = config["random_state"] 
    unique_experiment_path = config["unique_experiment_path"] 
    

    DATAPATH = PROJECT_FOLDER+"/data"
    experiment_name = datetime.datetime.now().strftime("%Y%m%d")
    experiment_path = f"Exp_{data_type}_{k_mer_size}_mer_{experiment_name}"
    
    if unique_experiment_path:
        experiment_path = unique_experiment_path
       
    if save_loacally==True:
        output_loc = f"{PROJECT_FOLDER}/models/experiments/{experiment_path}"
        print(f"Saving experiment loacally to: {output_loc}")
    else:
        output_loc = f"/content/{experiment_path}"
        print(f"Saving temporary to: {output_loc}")
        
    config["pretrained_model_path"] = pretrained_model_path
    config["experiment_path"] = experiment_path
    config["seq_length"] = seq_length
    config["k_mer_size"] = k_mer_size
    config["output_loc"] = output_loc
    config["random_state"] = random_state
    with open(config_path, "w") as outfile: 
        json.dump(config, outfile)
        
def read_config(): 
    """
    Read the config.json file and return as dictionary
    """ 
    with open(config_path) as f:
        config = json.load(f)
    if not os.path.isdir(config["output_loc"]):
        os.mkdir(config["output_loc"]) 
    return config