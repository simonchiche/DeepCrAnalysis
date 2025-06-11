import numpy as np
import os
import matplotlib.pyplot as plt
import subprocess
from datetime import datetime

def MatplotlibConfig(WorkPath, SimDir):

    #region Matplotlib config (build a function)
    font = {'family' : 'DejaVu Sans',
            'weight' : 'normal',
            'size'   : 14}

    plt.rc('font', **font)
    plt.rc('axes', titlesize=12)  
    #endregion

    #region Store configuration
    # We create a directory where the outputs are stored
    date = datetime.today().strftime('%Y-%m-%d')
    WorkPath = os.getcwd()
    OutputPath = WorkPath + "/Plots/" + SimDir + "/" + date + "/" 
    #print(OutputPath)
    #sys.exit()
    cmd = "mkdir -p " + OutputPath
    p =subprocess.Popen(cmd, cwd=os.getcwd(), shell=True)
    stdout, stderr = p.communicate()
    #endregion

    return OutputPath
