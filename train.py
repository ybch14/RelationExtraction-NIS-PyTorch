import os, sys, time
import importlib
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, "models"))
import utils

if __name__ == "__main__":
    model_name = utils.parse_args_name(sys.argv, "model")
    output_path = utils.parse_args_name(sys.argv, "output_path")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    OUTPUT_DIR = os.path.join(output_path, time.strftime("%Y%m%d%H%M", time.localtime()))
    if os.path.exists(OUTPUT_DIR):
        os.system("rm -rf %s" % OUTPUT_DIR)
    os.mkdir(OUTPUT_DIR)
    os.system("cp scripts/run_%s.sh %s" % (model_name, OUTPUT_DIR))
    os.system("cp models/%s.py %s" % (model_name, OUTPUT_DIR))
    PRFILE_PATH = os.path.join(OUTPUT_DIR, "prfile")
    os.mkdir(PRFILE_PATH)
    SNAPSHOT_PATH = os.path.join(OUTPUT_DIR, "snapshot")
    os.mkdir(SNAPSHOT_PATH)
    log_fp = open(os.path.join(OUTPUT_DIR, "log.txt"), 'w')
    logger = utils.Logger(log_fp)
    try:
        MODEL = importlib.import_module(model_name)
        MODEL.train(SNAPSHOT_PATH, PRFILE_PATH, verbose=True, logger=logger)
        log_fp.close()
    except KeyboardInterrupt:
        logger("Interrupted.")
        log_fp.close()
        os.system("rm -rf %s" % OUTPUT_DIR)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception as e:
        import traceback
        logger("Got exception.")
        log_fp.close()
        print(traceback.format_exc())
        os.system("rm -rf %s" % OUTPUT_DIR)
