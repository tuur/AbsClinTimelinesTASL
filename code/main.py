import argparse, os, sys, shutil, random, yaml, datetime
from lib.i2b2 import read_i2b2_folder
from lib.models.ATLM import *
from lib.data import Logger
from eval_script import eval_dirs
from lib.utils import get_phase_cfg
random.seed(0)

def main():
    #--------------------------------
    #  ARGUMENT HANDLING AND LOGGING
    #--------------------------------
    parser = argparse.ArgumentParser(description='Absolute Probabilistic Timeline Modeling')
    parser.add_argument('-config', type=str, default='./config.yml',
                        help='Config file (yaml format), default: ./config.yaml')
    parser.add_argument('-c', type=int, default=0,
                        help='Continue training from latest checkpoint.')
    parser.add_argument('-prefix', type=str, default='E-',
                        help='Adds a prefix to the model directory')
    args = parser.parse_args()

    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        torch.cuda.set_device(cfg['gpu'])

    exp_dir = "./out/" + datetime.datetime.now().strftime("%Y-%m-%d") + "/" + args.prefix + cfg["experiment_name"] + "/"
    print(args.config, exp_dir )
    print('Logging to', exp_dir)
    print('PID',os.getpid())

    t0 = time.time()
#    task = TimeLineTask(cfg)
#    for v in [CalenderPoint(1991,3,6,12,0), CalenderPoint(1991,3,9,12,00), CalenderPoint(1991,3,10,12,00),CalenderPoint(1991,3,11,12,0),CalenderPoint(1991,3,12,0,00), CalenderPoint(1991,3,12,9,0), CalenderPoint(1991,3,12,9,42), CalenderPoint(1991,3,12,9,43), CalenderPoint(1991,3,12,9,44),CalenderPoint(1991,3,12,9,45), CalenderPoint(1991,3,12,9,53),CalenderPoint(1991,3,12,10,0)]:
#        print(v, '\t', task.calender_point_to_minute_value(v))

    #--------------------------------
    #           TRAINING
    #--------------------------------

    if cfg['train_data_path']:
        print(cfg['train_data_path'])
        btime_train_data = read_i2b2_folder(cfg['train_data_path'], allow_zero_duration=False,keep_inconsistent_annotations=False,course=cfg['course_eval'])

        if args.c and os.path.exists(exp_dir):
            sys.stdout = Logger(stream=sys.stdout, file_name=exp_dir + '/train_log.log', log_prefix=str(args))
            if not args.c[-2:] == '.p':
                checkpoint_file_path = get_latest_checkpoint_from_dir(exp_dir + '/checkpoints')
            else:
                checkpoint_file_path = args.c
            model = load_model(checkpoint_file_path)
            exp_dir = exp_dir + '/c/'
            model.model_dir = exp_dir
        if os.path.exists(exp_dir):
            shutil.rmtree(exp_dir)
        os.makedirs(exp_dir)
        sys.stdout = Logger(stream=sys.stdout, file_name=exp_dir + '/train_log.log', log_prefix=str(args))
        if not args.c:
            model = eval(cfg['model_type'] + "(cfg, btime_train_data)")
            model.model_dir = exp_dir
        shutil.copy(args.config, exp_dir)
        for phase in range(len(cfg['phases'])):
            phase_cfg = get_phase_cfg(cfg, phase)
            model.train_model(phase_cfg, btime_train_data)

    #--------------------------------
    #           TESTING
    #--------------------------------

    if cfg['test_data_path']:
        sys.stdout = Logger(stream=sys.stdout, file_name=exp_dir + '/test_log.log', log_prefix=str(args))

        btime_test_data = read_i2b2_folder(cfg['test_data_path'], allow_zero_duration=False, keep_inconsistent_annotations=False,course=cfg['course_eval'])

        model = load_model(path=exp_dir + '/model.p')

        pred_dir = exp_dir + '/predictions/'
        predictions = model.pred_docs(btime_test_data, pred_dir)

        print(' -- Eval All --')
        evaluation = eval_dirs(cfg['test_data_path'], pred_dir,course=cfg['course_eval'])
        print(' -- Eval Bounded --')
        evaluation = eval_dirs(cfg['test_data_path'], pred_dir,course=cfg['course_eval'],has_timeml_timex_link=True)


    print('done in', time.time()-t0,'seconds')


if __name__ == '__main__':
    main()