import sys
import glob
import os
import subprocess
import multiprocessing


def run_this_sim(ymlpath):
    ymlfile_split = ymlpath.split("/")
    ymlfile = ymlfile_split[-1]
    ymldir = os.path.join(*ymlfile_split[:-1])
    ymltag = ymlfile.split(".")[0]
    logfile = os.path.join(ymldir,ymltag) + ".log"
    print("ymlpath=", ymlpath)
    print("logfile=", logfile)
    cmdline = ["python", "runSim.py", ymlpath]
    #print(cmdline)
    with open(logfile, "wb") as log:
        subprocess.run(cmdline, stdout=log, stderr=log)


if __name__ == "__main__":
    ymldir = sys.argv[1]
    if len(sys.argv) > 2:
        pool_size = int(sys.argv[2])
    else:
        pool_size = 10

    yml_search_str = os.path.join(ymldir, "*.yml")
    yml_list = glob.glob(yml_search_str)
    pool = multiprocessing.Pool(processes=pool_size)
    outputs = pool.map_async(run_this_sim, yml_list)
    pool.close()
    pool.join()
