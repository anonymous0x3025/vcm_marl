import os
import time
from run import run
from utils.param_utils import print_parameters
from multiprocessing import Process
import ray

COMPARISON = True
# COMPARISON = False


def main():
    try:
        from parameters import parameters
        params = parameters.Parameters()
        if params.DEVICE.find('cuda') > 0:
            ray.init(num_gpus=params.NUM_GPUS)
        else:
            ray.init(num_cpus=params.NUM_CPUS)
        print_parameters(params, params.param_overlapped_dict)

        for run_number in range(params.RUNS):
            run(params, run_number)
            print("Run:{}/{} is finished".format(run_number, params.RUNS))

        print("Exiting script")
    except KeyboardInterrupt:
        ray.shutdown()

    ray.shutdown()
    os._exit(os.EX_OK)


def comparison_main():
    try:
        from parameters import multi_parameters
        params_list = multi_parameters.MultiParameters().params_list
        RUNS = params_list[0].RUNS

        for run_number in range(RUNS):
            ps = [Process(target=run, args=(params, run_number)) for params in params_list]
            for p in ps:
                p.daemon = False
                p.start()
            for p in ps:
                p.join()
    except KeyboardInterrupt:
        time.sleep(1.0)
        ray.shutdown()
    finally:
        print("Exiting script")
        time.sleep(1.0)
        ray.shutdown()
        # Making sure framework really exits
        os._exit(os.EX_OK)


if __name__ == '__main__':
    if COMPARISON:
        comparison_main()
    else:
        main()
