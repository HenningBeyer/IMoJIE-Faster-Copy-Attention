import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 

def PR_curve(dir, from_model_dirs, name, models=['gru','lstm']):
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 0.5])   
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    line_styles = ['-','--','-.',':']
    for mod in from_model_dirs:
        
        fig, axs = plt.subplots(1,1,figsize=(10,10))
        for _, __, files in os.walk(dir + mod): #Erhalte den Dateinamen
            print(files)
            for filename in files:
                if 'pro_output' in filename:
                    file = filename
                    
        #for i in range(len(models)):      #Erhalte für jedes Modell ein eigenes Linienmuster
        #    if dir[:dir.find('_')] == models[i]:
        #        line_style = line_styles[i]
        
        lines = open(dir + mod + '/' + file, 'r').readlines()
        for i in range(len(lines)):
            lines[i] = lines[i][:-1].split('\t')
        lines = np.array([(lines[1:])]).squeeze()
        lines = np.delete(lines, 2, axis=1)       
        prec = lines[:, 0].astype(np.float64)
        rec = lines[:, 1].astype(np.float64) 
        axs.plot(rec, prec, label = mod.replace('_', ' '), linewidth=1)
        
  
    plt.legend(loc="lower right")
    fig.savefig(name, dpi=12*24)
    plt.plot(rec, prec, label = mod.replace('_', ' '), linewidth=1)
    plt.show()

def main():
    print(os.getcwd())
    dir = 'G:/Next projects/Bell/Code/ablage/PR-plot/'
    PR_curve(dir=dir, from_model_dirs=['gru_bs128','gru_bs8_cd0', 'lstm_bs16', 'lstm_bs8_cd0'], name='PR_curve_best.png') 
  
  
main()  
