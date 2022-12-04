import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
 
def save_stripped(dn, file_lines): 
    with open(dn + 'stripped.txt', 'w') as file:
        for i in range(len(file_lines)):
            l = file_lines[i][file_lines[i].find('[[')+2: file_lines[i].find(']]')] #mit start-tokens
            file.write(l + '\n')
               
def save_sent_count(dn, file_lines):  #ein Satz sollte aus folgender Struktur bestehen [1,2,3,4,5,6] = ['<arg1>', '</arg1>', '<rel>', '</rel>', '<arg2>', '</arg2>']
    #(Sätze aus nur Padding-Tokens werden NICHT mitgezählt; nur richtige Extraktionen werden gezählt)
    with open(dn + 'sentcount.txt', 'w') as file:
        for i in range(len(file_lines)): 
            l = file_lines[i][file_lines[i].find('[[')+2: file_lines[i].find(']]')]  #nur die token-IDs aus der Textdatei entnehmen & start-token entfernen
            count = l.count(' 102, 1,') + 1 #einzelne Tokens werden wiederholt gerneriert --> [' 102, 1,']
                                            #erstes ' 102, 1,' wird nicht gezält --> +1
            file.write(str(count) + '\n')  
 
def save_word_count(dn, file_lines, count_padding=True, count_special=True, count_EOE=True):
    sent_counts = np.array(open(dn + 'sentcount.txt', 'r').readlines()).astype(np.int16)
    
    mtbsc = max_testbatch_sent_counts = np.array([]) # je nach maximaler Anzahl an Extraktionen wird pro Extraktion ein Token hinzugefügt
    for i in range(5):
        mtbsc = np.append(mtbsc, [np.max(sent_counts[i*128:(i+1)*128])] * 128 ) # der Maximal-wert des Test-Batches wird 128 mal hinzugefügt
    mtbsc = np.append(mtbsc, sent_counts[-1])
   
    with open(dn + 'wordcount.txt', 'w') as file:
        for i in range(len(file_lines)): 
            l = file_lines[i][file_lines[i].find('[[')+2: file_lines[i].find(']]')]  #nur die token-IDs aus der Textdatei entnehmen & start-token entfernen
            if not count_padding:      
                l = l[:-5].replace(' 102,', '') # remove all '102, ' 
                if l[len(l)-5:] == ', 102':
                    l = l[:-5] #remove last padding 
            if not count_special:
                for i in range(9):  #spezielle Tokens: Token-ID 1-9                                
                    l = l.replace(' ' + str(i+1)+',', '')
            if not count_EOE:
                l = l.replace(' 10,', '')
                if l[len(l)-4:] == ', 10': #remove last EOE-Tag
                    l = l[:-4]               
            file.write(str(l.count(',') + 1 - int(mtbsc[i])) + '\n') #count of ',' + 1 = Anzahl der Tokens; -sentcounts da letzter Padding-Token außerhalb der Extrahierung angefügt wird 
 
def get_specific_wordcounts(extr_line): #eine Extraktion (extr_line) beinhaltet mitunter 10 Sätze
    l = extr_line.split(', ')
    wordcount_dict = {unique : l.count(unique) for unique in np.unique(l)}
    return wordcount_dict 
  
def save_extr_repetitions(dn, file_lines, count_not_wrong_sents=True, count_not_special_repetitions=False): #für jeden Satz mit wiederholungsproblem wird eine 1 gespeichert
    """ Beispiele von Tokenwiederholungen:
        [1, 2238, 139, ...,  20080, 15012, 2572, 2572, 2572, 2572, 2572, ..., 102, 1, ...]  
        [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 102, 1, ...]
        [1, 24898, 1, 24898, 1, 2, 24898, 2, 24898, 2, 24898, 2, 24898, 2, 24898, 2,  ...]
    """     
    with open(dn + 'extraction_repetitions.txt', 'w') as file: 
        mwps = max_wordcount_per_sanetence = 9  #deckt sehr präzise über 99,3% alle Fälle ab
        for i in range(len(file_lines)): 
            l = file_lines[i][file_lines[i].find('[[')+2: file_lines[i].find(']]')]
            sents = np.array(l.split(', 102, 1,'))
            #print(len(sents)) 
            for s in sents: 
                bool = False               
                wordcount_dict = get_specific_wordcounts(s)
                if '102' in wordcount_dict:
                    wordcount_dict.pop('102')    #Padding-Token sind bis zu 300 mal vertreten
                # if not count_special:
                #     for j in range(6):
                #         st = str(j+1)
                #         if st in wordcount_dict.keys():
                #             wordcount_dict.pop(st) #entfernt Tokens 1-6
                if '10' in wordcount_dict and mwps <= 10: 
                    wordcount_dict.pop('10')   
                max_val = max(wordcount_dict.values())

                if count_not_wrong_sents:  #Quelle imojie_master/benchmark/oie-readers/allennlpReader.py 
                                          #filtert sätze heraus, die auch durch den allennlp reader herausgefiltert werden und also nicht gewertet werden 

                    s = ' 1, ' + s # '1' wurde durch .split() entfernt 

                    try:
                        arg1 = s[s.index(' 1,'):s.index(' 2,')]
                    except:
                        arg1 = ""
                    try:
                        rel = s[s.index(' 3,'):s.index(' 4,')]
                    except:
                        rel = ""
                    try:
                        arg2 = s[s.index(' 5,'):s.index(' 6,')]
                    except:
                        arg2 = "" 

                    if not ( arg1 or arg2 or rel):
                        bool = True
                if count_not_special_repetitions:
                    for j in range(6): 
                        key = str(j+1)   
                        if key in wordcount_dict:
                           if  wordcount_dict[key] >= mwps:  #Zählt bei Wortwiederholung der Tokens <arg1>, ..., </arg2> keine Wortwiederholungen. 
                               bool = True                   
                if bool:                                      
                    continue
                                            
                if max_val >= mwps and max_val < 15:
                    file.write(f'token:{max(wordcount_dict, key=wordcount_dict.get)} count: {max_val}; Line: {i+1}:    {s}' + '\n') #dokumntiert unsichere Wiederholungen
                elif max_val >= mwps:
                    file.write(f'token:{max(wordcount_dict, key=wordcount_dict.get)} count: {max_val}; Line: {i+1}:' + '\n')  

def make_folders(model_names=['gru','lstm']):
    for dn in get_all_dirnames(model_names):
        if not os.path.exists(dn):
            os.mkdir(dn)
    
def get_all_dirnames(model_names=['gru','lstm']):
    files = []
    for s in model_names:
        files = files + [s + '_bs' + str(8*2**i) for i in range(4,-1,-1) ] + [s +'_bs8_cd0']  # bsp lstm_bs128, lstm_bs64, ... lstm_bs8_cd0
    return files

def get_output_file(dir):
    for _, __, files in os.walk(dir):
        for file in files:
            if file[len(file)-6:] == '.jsonl':
                return file
        raise(FileNotFoundError)
 
 
 
def get_all_batch_stats(model_dirs, models=['gru', 'lstm']): #['gru_bs128', 'gru_bs64', 'gru_bs32', ...,  'lstm_bs8', 'lstm_bs8_cd0']
    #erhalte Spaltenwerte 
    model_dict = {}
    for mod in models:
        model_dict.update({mod : {}}) 
        model_dict.get(mod).update({'Ø max Sätze' : np.array([]),'Ø min Sätze' : np.array([]),
                                        'Ø Sätze' : np.array([]),'Ø Tokens' : np.array([]),
                                        'Σ Sätze' : np.array([]),'Σ Tokens' : np.array([]), 
                                    'Σ Tokenwiederholungen' : np.array([])})#, #Eine Extraktion beinhaltet bis zu Zehn extrahierte Sätze
                                   ### 'Anteil der Sätze mit Tokenwiederholungen': np.array([])}) # 100% --> in jeder Extraktion ist jeder Satz von Wortwiederholung betroffen
    
    def append_model_values_to_model_dict(model_dict_key, np_function, dir_lines):       
        vals = model_dict.get(mod).get(model_dict_key)                
        for j in range(5): #CaRB hat 641 Testsätze und benötigt somit 5+1 trainingsbatches bei Batch-Size 128 
            vals = np.append(vals,np_function(dir_lines[j*128 : (j+1)*128]))  
        vals = np.append(vals, np_function(dir_lines[641-1]))
        model_dict.get(mod)[model_dict_key] = vals          
        
    for mod_dir in model_dirs:
        for mod_ in models:  #bestimme Modell ['gru', 'lstm'] anhand von Verzeichnis
            if (mod_dir.find(mod_) > -1):
                mod = mod_         
        #Dateien auslesen
        sent_counts = open(mod_dir +'/sentcount.txt', 'r').readlines()
        sent_counts = np.array([int(s[:-1]) for s in sent_counts])  # '5\n' --> 5
        
        word_counts = open(mod_dir +'/wordcount.txt', 'r').readlines()
        word_counts = np.array([int(s[:-1]) for s in word_counts])  # '5\n' --> 5
    
        append_model_values_to_model_dict('Ø max Sätze', np.max, sent_counts)  # kein extra np.mean gebraucht (Einzelwert)
        append_model_values_to_model_dict('Ø min Sätze', np.min, sent_counts)
        append_model_values_to_model_dict('Ø Sätze', np.mean, sent_counts)
        append_model_values_to_model_dict('Ø Tokens', np.mean, word_counts)
        append_model_values_to_model_dict('Σ Sätze', np.sum, sent_counts)
        append_model_values_to_model_dict('Σ Tokens', np.sum, word_counts) 

        rep_lines = open(mod_dir +'/extraction_repetitions.txt', 'r').readlines()
        rep_lines = np.unique([int(l.split(':')[3][1:]) for l in rep_lines])   # 'token:2 count: 48; Line: 316: ... \n' --> 316   (erhält Zeilen mit Tokenwiederholung)
        
        vals = model_dict.get(mod).get('Σ Tokenwiederholungen')
        batch_ext_rep_counts = [0,0,0,0,0,0] # --> liste [68, 65, 73, 65, 60, 1] mit Anzahl an Extraktionen mit Wiederholungen
        for i in range(5):
            batch_ext_rep_counts[i] = sum((i*128 < rep_lines) & (rep_lines <= i*128 + 128))                   
        batch_ext_rep_counts[5] = 1 if rep_lines[-1] == 641 else 0                        
        vals = np.append(vals, batch_ext_rep_counts)  
        model_dict.get(mod)['Σ Tokenwiederholungen'] = vals
        
           
    return model_dict


def get_combined_batch_stats(model_dict, models=['gru', 'lstm'], seperate_sixth_batch=False): #['gru_bs128', 'gru_bs64', 'gru_bs32', ...,  'lstm_bs8', 'lstm_bs8_cd0']
    
    def combine_all(model_dict_key, mod, np_func): #Für kleine Tabelle
        arr = model_dict.get(mod).get(model_dict_key)  
        model_dict.get(mod)[model_dict_key] = np.array([np_func(arr[i*6:i*6+6]) for i in range(6)]) 
        
    def combine_with_extrabatch(model_dict_key, mod, np_func): #Für mittlere Tabelle
        arr = model_dict.get(mod).get(model_dict_key)  
        model_dict.get(mod)[model_dict_key] = np.array([np.append(np_func(arr[i*6:i*6+5]), arr[i*6+5]) for i in range(6)]).flatten()
        
    for mod in models:
        if not seperate_sixth_batch:           
            combine_all('Ø max Sätze', mod, np.mean)    #np.mean fasst alle maximalen Werte zusammen
            combine_all('Ø max Sätze', mod, np.mean)
            combine_all('Ø Sätze', mod, np.mean)
            combine_all('Ø Tokens', mod, np.mean)
            combine_all('Σ Sätze', mod, np.sum)
            combine_all('Σ Tokens', mod, np.sum)
            combine_all('Σ Tokenwiederholungen', mod, np.sum)
        else: 
            combine_with_extrabatch('Ø max Sätze', mod, np.mean)
            combine_with_extrabatch('Ø min Sätze', mod, np.mean)
            combine_with_extrabatch('Ø Sätze', mod, np.mean)
            combine_with_extrabatch('Ø Tokens', mod, np.mean)
            combine_with_extrabatch('Σ Sätze', mod, np.sum)
            combine_with_extrabatch('Σ Tokens', mod, np.sum)
            combine_with_extrabatch('Σ Tokenwiederholungen', mod, np.sum)
                                        
    return model_dict
  
            
def full_table(model_dict, models=['gru','lstm']):
    #Erstelle Multi-Reihen-index
    train_batches = np.append(np.array([[8*2**i]*6 for i in range(5)])[::-1].flatten(), [8,8,8,8,8,8]) 
    testbatches = np.array([[1,2,3,4,5,6]*6]).flatten()
    arr = np.stack((train_batches,testbatches))
    index_ = pd.MultiIndex.from_arrays(arr, names=["Train-Batch-Size", "Testbatch"])
    
    #Erstelle Multi-Spaltenindex
    mc = model_count = len(models)
    col_index1 = np.array([['Ø max Sätze']*mc,['Ø Sätze']*mc,['Ø Tokens']*mc,['Σ Sätze']*mc,['Σ Tokens']*mc, ['Σ Tokenwiederholungen']*mc]).flatten()
    col_index2 = np.array(models*6)
    arr = np.stack((col_index1,col_index2))
    col_index = pd.MultiIndex.from_arrays(arr)

    #Ordne Spalten in array ein
    first = True
    for col in col_index1[::mc]: # Ø max Sätze, Ø Sätze, Ø Tokens, Σ Sätze, Σ Tokens
        for mod in models:     
            if first:
                arr = np.array([model_dict[mod][col]])
                first = False
            else:
                arr_ = np.array([model_dict[mod][col]])
                arr = np.vstack((arr,arr_))
    df = pd.DataFrame(arr.swapaxes(0,1), index=index_, columns=col_index)
    df = df.round(1)  
    #df = df.drop(columns=['Ø max Sätze'])
    print(df)
    df.to_excel('Große_Tabelle.xlsx')


def medium_table(model_dict, models_dirs, models=['gru','lstm']): #separiert Trainbatches 1-5 und 6
    #Erstelle Reihen-index
    index_ = np.append(np.array([[8*2**i] for i in range(5)])[::-1].flatten(), [8]) 
    
    train_batches = np.append(np.array([[8*2**i]*2 for i in range(5)])[::-1].flatten(), [8,8]) 
    testbatches = np.array(['1-5', '6']*6)
    arr = np.stack((train_batches,testbatches))
    index_ = pd.MultiIndex.from_arrays(arr, names=["Train-Batch-Size", "Testbatch"]) 
    
    #Erstelle Multi-Spaltenindex
    mc = model_count = len(models)
    col_index1 = np.array([['Ø max Sätze']*mc,['Ø Sätze']*mc,['Ø Tokens']*mc,['Σ Sätze']*mc,['Σ Tokens']*mc, ['Σ Tokenwiederholungen']*mc]).flatten()
    col_index2 = np.array(models*6)
    arr = np.stack((col_index1,col_index2))
    col_index = pd.MultiIndex.from_arrays(arr)

    #Ordne Spalten in array ein
    first = True
    for col in col_index1[::mc]: # Ø max Sätze, Ø Sätze, Ø Tokens, Σ Sätze, Σ Tokens
        for mod in models:     
            if first:
                arr = np.array([model_dict[mod][col]])
                first = False
            else:
                arr_ = np.array([model_dict[mod][col]])
                arr = np.vstack((arr,arr_))
    df = pd.DataFrame(arr.swapaxes(0,1), index=index_, columns=col_index)
    df = df.round(1) 
    df = df.drop(columns=['Ø max Sätze', 'Σ Sätze']) 
    print(df)
    df.to_excel('Mittlere_Tabelle.xlsx')   


def small_table(model_dict, models_dirs, models=['gru','lstm']):
    #Erstelle Reihen-index
    index_ = np.append(np.array([[8*2**i] for i in range(5)])[::-1].flatten(), [8])  
    
    #Erstelle Multi-Spaltenindex
    mc = model_count = len(models)
    col_index1 = np.array([['Ø max Sätze']*mc,['Ø Sätze']*mc,['Ø Tokens']*mc,['Σ Sätze']*mc,['Σ Tokens']*mc, ['Σ Tokenwiederholungen']*mc]).flatten()
    col_index2 = np.array(models*6)
    arr = np.stack((col_index1,col_index2))
    col_index = pd.MultiIndex.from_arrays(arr)

    #Ordne Spalten in array ein
    first = True
    for col in col_index1[::mc]: # Ø max Sätze, Ø Sätze, Ø Tokens, Σ Sätze, Σ Tokens
        for mod in models:     
            if first:
                arr = np.array([model_dict[mod][col]])
                first = False
            else:
                arr_ = np.array([model_dict[mod][col]])
                arr = np.vstack((arr,arr_))
    df = pd.DataFrame(arr.swapaxes(0,1), index=index_, columns=col_index)
    df = df.rename_axis('Train-Batch-Size')
    df = df.round(1) 
    df = df.drop(columns=['Ø max Sätze', 'Σ Sätze'], axis=1) 
    print(df)
    df.to_excel('Kleine_Tabelle.xlsx')
   
   
    
def PR_curve(from_model_dirs, name, models=['gru','lstm']):
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 0.5])   
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    line_styles = ['-','--','-.',':']
    for dir in from_model_dirs:
        
        for _, __, files in os.walk(dir): #Erhalte den Dateinamen
            for filename in files:
                if 'pro_output' in filename:
                    file = filename
                    
        for i in range(len(models)):      #Erhalte für jedes Modell ein eigenes Linienmuster
            if dir[:dir.find('_')] == models[i]:
                line_style = line_styles[i]
        
        lines = open( dir + '/' + file, 'r').readlines()
        for i in range(len(lines)):
            lines[i] = lines[i][:-1].split('\t')
        lines = np.array([(lines[1:])]).squeeze()
        lines = np.delete(lines, 2, axis=1)       
        prec = lines[:, 0].astype(np.float64)
        rec = lines[:, 1].astype(np.float64) 
        plt.plot(rec, prec, label = dir.replace('_', ' '), linewidth=1, ls=line_style)
        
  
    plt.legend(loc="lower right")
    plt.savefig(name, dpi=12*24)
    plt.show()

def save_tokencount_of_all_subextractions(dn, file_lines): 
    
    with open(dn + 'subextraction_tokencounts.txt', 'w') as file:
        for i in range(len(file_lines)): 
            sent_token_counts = [0,0,0,0,0,0,0,0,0,0]
            l = file_lines[i][file_lines[i].find('[[')+2: file_lines[i].find(']]')]  #nur die token-IDs aus der Textdatei entnehmen & start-token entfernen
            l = l.split(' 102, 1,') #entfernt zwei Tokens bzw. Kommas  
            for j in range(len(l)): #Es müssen richtige und leere Extraktioen mit nur Padding getrennt werden
                if ', 10,' in l[j]:
                    l[j] = l[j].split(', 10,')   #fügt durch split Listen in die Liste l mit den geteilten Strings ein
            split_l = []
            for item in l: #konvertiert bei 2 listen in eine Liste
                if type(item) == list:
                    for item_ in item:
                        if len(item_.split(',')) < 6:  #--> Falscher Split bei richtgier extraktion mit EOE-Token, tritt vor allem of bei Tokenwiederholungen mit Token [10] auf
                                                #Fehler ansonsten: '1,13,2,3,56,43,4,5,67,43,6,102,102,10,102,102' --> ['1,13,2,3,56,43,4,5,67,43,6,102,102', '102,102']
                                                #Es gibt noch weitere Unregelmäßigkeiten Bsp 'lstm_bs8/subextraction_tokencounts.txt' 
                                                #Anstatt mit weiteren Code diese sehr kleinen Probleme zu beheben, nutze ich den am meisten vertretenen Tokencount für die Tabellen
                            split_l[-1] += ', 10,' + item_
                        else:
                            split_l.append(item_)                     
                else:
                   split_l.append(item) 
            l = split_l
            for j in range(10):
                if j > len(l)-1: #Falls schon über alle Sätze der Extraktion iteriert: Abbruch
                    break               
                sent_token_counts[j] = l[j].count(',')  
                if j == 0:  #beim ersten Satz (falls kein weiterer folgt)
                    sent_token_counts[j] += 1    
                elif 0 < j <= len(l)-1:   #bei allen Sätzen die nicht die erste Teilextraktionen sind (auch für letzen Satz)
                    sent_token_counts[j] += 2 
            file_str = '' 
            for j in range(10):
                if sent_token_counts[j] > 0:
                    sent_token_counts[j] -= 1    #Es wird ein Token zu viel pro Satz generiert. Dieser stammt nicht vom neuralen Modell und sagt somit nichts über die Geschwindigkeit aus
                file_str += str(sent_token_counts[j]) + '\t'
            file.write(file_str + '\n') 
            
            
def get_trainbatch_subextraktions(dn): # gibt 
    lines = np.array(open(dn + 'subextraction_tokencounts.txt', 'r').readlines())
    lines = np.array([l.split('\t')[:-1] for l in lines])
    lines = lines.astype(np.int16)
    arr = np.array([])
    for i in range(5):
        bl = batch_lines = lines[i*128:128*(i+1)]
        for j in range(10):
            c = np.bincount(bl[:,j]).argmax()
            arr = np.append(arr,c)     #jede extraktion im selben Testbatch hat die selbe Anzahl an Tokens --> die am meisten gezählte Anzahl ist die richtige
    arr = np.append(arr, lines[641-1])        
    return arr

def table_subextraction_tokencount(dirs, models=['gru', 'lstm']):  #jede Extraktion beeinhaltet bis zu zehn Sätze, bei denen hierbei für jeden der zehn Sätze die Anzahl der Tokens zurückgegeben wird 
    #Erstelle Multi-Reihen-index
    train_batches = np.append(np.array([[8*2**i]*6 for i in range(5)])[::-1].flatten(), [8,8,8,8,8,8]) 
    train_batches = np.append(train_batches, train_batches)  # da 2 Modelle separat
    testbatches = np.array([[1,2,3,4,5,6]*6]).flatten()
    testbatches = np.append(testbatches, testbatches)  # da 2 Modelle separat
    model_index = np.array([['gru']*(len(train_batches) // 2), ['lstm']*(len(train_batches) // 2)]).flatten()
    arr = np.stack((model_index, train_batches,testbatches))
    index_ = pd.MultiIndex.from_arrays(arr, names=["Modelle","Train-Batch-Size", "Testbatch"])
    
    #Erstelle Multi-Spaltenindex
    mc = model_count = len(models)
    sent_indexes = [str(i) for i in range(1,11)]
    col_index1 = np.array([sent_indexes]).flatten()  
     
    vals = np.array([])  
    for dir in dirs:
        vals = np.append(vals, get_trainbatch_subextraktions(dir + '/'))
    arr = np.array([])
    first = True
    for i in range(int(len(vals)/10)):
        arr_ = np.array([vals[10*i:10*(i+1)]])
        if not first:
            arr = np.vstack([arr, arr_])
        else:
            arr = arr_
            first = False
                           
    df = pd.DataFrame(arr, index=index_, columns=col_index1)
    print(df)
    df.to_excel('Tokenanzahl_der_Teilextraktionen.xlsx')   
     

def main(prepare_folders=True, analyse=True, pr_curves=True):
    models = ['gru','lstm'] 
    dirs = get_all_dirnames(models) 
    print(dirs)  
    i = 0
    if prepare_folders:
        make_folders()
        for dn in dirs:
            i += 1                  
            file_lines = open(dn + '/' + get_output_file(dn + '/'), 'r').readlines()
            save_stripped(dn + '/', file_lines)
            save_sent_count(dn + '/', file_lines) 
            save_word_count(dn + '/', file_lines, count_padding=True, count_EOE=True) #Zählt hier alle Tokens           
            save_extr_repetitions(dn + '/', file_lines, count_not_wrong_sents=True)
            save_tokencount_of_all_subextractions(dn + '/', file_lines)
            print('prepared ' + str(i) + '/' + str(len(dirs)) + ' folders')
            
    if analyse:
        model_dict = get_all_batch_stats(dirs)
        full_table(model_dict, models)
        
        model_dict = get_combined_batch_stats(model_dict, models, seperate_sixth_batch=True)
        medium_table(model_dict, models)
        
        model_dict = get_all_batch_stats(dirs)
        model_dict = get_combined_batch_stats(model_dict, models, seperate_sixth_batch=False)
        small_table(model_dict, models)
        
        table_subextraction_tokencount(dirs, models=models)
        
        if pr_curves:   
            PR_curve(from_model_dirs=dirs, name='PR_curve_all.png', models=models) 
            PR_curve(from_model_dirs=['gru_bs128','gru_bs8_cd0', 'lstm_bs16', 'lstm_bs8_cd0'], name='PR_curve_best.png') 
  
  
main(prepare_folders=True, analyse=True, pr_curves=False)        