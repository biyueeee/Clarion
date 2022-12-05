
import numpy as np
import argparse
import os, sys, re
import joblib
import pandas as pd
root_path = os.getcwd()
# print(root_path)

def read_fasta(inputfile):
    if os.path.exists(inputfile) == False:
        print('Error: file " %s " does not exist.' % inputfile)
        sys.exit(1)
    with open(inputfile) as f:
        record = f.readlines()
    if re.search('>',record[0]) == None:
        print('Error: the input file " %s " must be fasta format!' % inputfile)
        sys.exit(1)

    data = {}
    count = 0
    for line in record:
        if line.startswith('>'):
            name = line.replace('>', '').split('\n')[0]
            data[name] = ''
        else:
            if len(line) <= 3000:
                data[name] += line.replace('\n', '')
            else:
                count += 1
                line = line.replace('\n', '')
                line_cut = line[:3000] +line[-3000:]
                data[name] += line_cut

    if count > 1:
        print('Warning: some fasta sequences will be truncated less than 6000 nt.')

    return data






def kmer_record(k):  # calculate kmer of k from 1 to 6

    base = ['A', 'C', 'G', 'T']
    _1mer = {}
    _2mer = {}
    _3mer = {}
    _4mer = {}
    _5mer = {}
    _6mer = {}

    for n1 in base:
        _1mer[n1] = 0
        for n2 in base:
            _2mer[n1 + n2] = 0
            for n3 in base:
                _3mer[n1 + n2 + n3] = 0
                for n4 in base:
                    _4mer[n1 + n2 + n3 + n4] = 0
                    for n5 in base:
                        _5mer[n1 + n2 + n3 + n4 + n5] = 0
                        for n6 in base:
                            _6mer[n1 + n2 + n3 + n4 + n5 + n6] = 0
    if k == 1:
        record = _1mer
    elif k == 2:
        record = _2mer
    elif k == 3:
        record = _3mer
    elif k == 4:
        record = _4mer
    elif k == 5:
        record = _5mer
    elif k == 6:
        record = _6mer
    else:
        print('Error: Please must be sure that k value is between 1 and 6.')

    return record




def kmer(seq,k):
    seq = seq.replace('U','T')
    re = kmer_record(k)
    total = 0
    for i in range(0,len(seq)):
        if i > (len(seq) - k):
            break
        else:
            tmp = seq[i : (i + k)]
            if tmp in re:
                re[tmp] += 1
            total += 1

    kmer_fea = []
    for i in re.values():
        ave = i / total
        kmer_fea.append(ave)

    return kmer_fea





def extrac_feature(input_seqs):
    x = []
    for seq in input_seqs.values():
        tmp1 = kmer(seq, 1)
        tmp2 = kmer(seq, 2)
        tmp3 = kmer(seq, 3)
        tmp4 = kmer(seq, 4)
        tmp5 = kmer(seq, 5)
        tmp6 = kmer(seq, 6)
        tmp = tmp1 + tmp2 + tmp3 + tmp4 + tmp5 + tmp6
        x.append(tmp)

    x = np.array(x)
    return x



def predict_SL(inputfile,outputfile):
    all = read_fasta(inputfile)
    feas = extrac_feature(all)
    clf = joblib.load('/var/www/html/Clarion/Clarion_main/'+'model')
    labels = clf.predict(feas,weight=0.65)
    # print(labels) #,dtype=bool
    labels = pd.DataFrame(labels,index=all.keys(),dtype=bool,columns=['Exosome','Nucleus','Nucleoplasm','Chromatin','Cytoplasm','Nucleolus','Cytosol','Membrane','Ribosome'])
    labels.to_csv(outputfile+'.txt',index_label='ID')








def main():
    parser = argparse.ArgumentParser(description='Clarion: a predictor to identify potential mRNA subcellular localizations')
    parser.add_argument('-input',dest='inputfile',type=str, required=True,
                        help='Query mRNA sequences in fasta format')
    parser.add_argument('-output',dest='outputfile',type=str,required=False,
                        help='The path where you want to save the prediction results')
    args = parser.parse_args()

    inputfile = args.inputfile
    outputfile = args.outputfile

    if outputfile != None:
        predict_SL(inputfile,outputfile)
        # print('Prediction results have been saved in '+outputfile)
    else:
        default_output = 'result'
        predict_SL(inputfile, default_output)
        # print('Prediction results have been saved in the current dictionary')



if __name__ == "__main__":
    main()














