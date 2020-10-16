
import argparse, os, shutil, yaml, glob
from copy import deepcopy


#lalalala

def main():
    #--------------------------------
    #  ARGUMENT HANDLING AND LOGGING
    #--------------------------------

    parser = argparse.ArgumentParser(description='Script to set up experiments')
    parser.add_argument('-config', type=str, default='./config.yml',
                        help='Config file (yaml format), default: ./config.yaml')
    parser.add_argument('-configs', type=str, default=None,
                        help='Config file (yaml format), default: ./config.yaml')
    parser.add_argument('-name', type=str, default='experiment',
                        help='Experiment name')
    parser.add_argument('-p', type=str,
                        help='Parameter to change')
    parser.add_argument('-vs', type=str,
                        help='Values to be assigned (split by a : )')
    parser.add_argument('-go', type=str,
                        help='Run in parrallel')
    parser.add_argument('-prefix', type=str,
                        help='Add a prefix to all experiment names', default="E")
    args = parser.parse_args()

    if args.configs:
        for config_path in glob.glob(args.configs +'*.yml'):
            with open(config_path, 'r') as ymlfile:
                cfg = yaml.load(ymlfile)
                cfg['experiment_name'] = args.prefix + '-' + cfg['experiment_name']
            exec_str = "python main.py -config " + config_path + ' -prefix '+ args.prefix + (' &' if args.go else '')
            print('exec:',exec_str)

            os.system(exec_str)
        exit()


    exp_dir = 'configs/' + args.name
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir)

    with open(args.config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    for v in args.vs.split(':'):
        new_config_path = exp_dir + "/" + args.p + "=" + v +'.yml'
        new_exp_name = args.name + "-" + args.p + "=" + v

        try:
            v = int(v)
        except:
            try:
                v = float(v)
            except:
                pass
        new_cfg = deepcopy(cfg)

        new_cfg[args.p] = v
        new_cfg['experiment_name'] = new_exp_name

        with open(new_config_path, 'w') as f:
            yaml.dump(new_cfg, f)
            print(args.p, cfg[args.p], v, new_config_path)

        exec_str = "python main.py -config " + new_config_path + (' &' if args.go else '')
        os.system(exec_str)




if __name__ == '__main__':
    main()