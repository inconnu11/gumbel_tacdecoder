import os
import pickle
import numpy as np

rootDir = 'assets/spmel'
f0Dir = 'assets/raptf0'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

speakers_list =sorted(subdirList) 
speakers = []
val = []
test = []
for speaker in speakers_list:
    print('Processing speaker: %s' % speaker)
    utterances = []
    #utterances.append(speaker)
    validations = []
    validations.append(speaker)
    tests = []
    tests.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion
    spkid = np.zeros((len(speakers_list),), dtype=np.float32)
    spkid[speakers_list.index(speaker)]=1.0
    #utterances.append(spkid)
    validations.append(spkid)
    tests.append(spkid)

    # create file list
    counter=0
    for fileName in sorted(fileList):
        if counter < len(fileList)*0.8:
            temp = []
            temp.append(speaker)
            temp.append(spkid) 
            temp.append(os.path.join(speaker,fileName))
            utterances.append(temp)
        elif counter < len(fileList)*0.9:
            #validation

            temp = []
            f0_num=np.load(os.path.join(f0Dir,os.path.join(speaker,fileName)))
            if len(f0_num)>400:
                continue
            temp.append(np.load(os.path.join(rootDir,os.path.join(speaker,fileName))))
            
            temp.append(np.load(os.path.join(f0Dir,os.path.join(speaker,fileName))))
            temp.append(len(f0_num))
            temp.append('Val_'+speaker+'_'+fileName)
            validations.append(temp)
            val.append(validations)

            validations=[]
            
            validations.append(speaker)
            validations.append(spkid)
        elif counter < len(fileList):
            temp = []
            temp.append(np.load(os.path.join(rootDir,os.path.join(speaker,fileName))))
            f0_num=np.load(os.path.join(f0Dir,os.path.join(speaker,fileName)))
            temp.append(np.load(os.path.join(f0Dir,os.path.join(speaker,fileName))))
            temp.append(len(f0_num))
            temp.append('test_'+speaker+'_'+fileName)
            tests.append(temp)
            

            test.append(tests)
            
            tests = []
            tests.append(speaker)
            tests.append(spkid)
        counter=counter+1
    speakers= speakers+utterances
    

a = test[0]
b = test[-2]   
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)   
with open(os.path.join(rootDir, 'validate.pkl'), 'wb') as handle:
    pickle.dump(val, handle)   
with open(os.path.join(rootDir, 'test.pkl'), 'wb') as handle:
    pickle.dump(test, handle)   