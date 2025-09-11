from jh_ml_models.model_fitting.hyper_parameter_tuner import HyperParameterTuner
from jh_ml_models.model_fitting.hyper_parameter_tuner import HyperParameter
import datetime
from datetime import datetime, timedelta
if __name__ == "__main__":
    finish_time = datetime.now()
    finish_time += timedelta(minutes=45)
    hyper_parameter_tuner = HyperParameterTuner(mode="tcn", epochs_per_run=300)

    hyper_parameter_dictionary = {
        # Common hps
        "lr" : HyperParameter(lower_limit=None, upper_limit=None, data_type=list, data_list=[1e-5, 1e-4, 1e-3, 1e-2]),
        "batch_size" : HyperParameter(lower_limit=None, upper_limit=None, data_type=list, data_list=[16, 32, 64, 128, 256]),
        # GRU
        "hidden_size" : HyperParameter(lower_limit=None, upper_limit=None, data_type=list, data_list=[16, 32, 64, 128, 256, 512, 1024]),
        "num_layers" : HyperParameter(lower_limit=1, upper_limit=10, data_type=int),
        # TCN
        "kernel_size" : HyperParameter(lower_limit=2, upper_limit=6, data_type=int),
        "C1_out_channels" : HyperParameter(lower_limit=1, upper_limit=24, data_type=int)
    }


    hyper_parameter_tuner.random_search(hyper_parameter_dictionary=hyper_parameter_dictionary, finish_time=finish_time)

