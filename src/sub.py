import pickle
from tqdm import tqdm
import numpy as np

def main():
    pht=np.zeros((2**17,1))
    ghr=0
    max_ghr=2**17-1
    total_count=0
    right_count=0
    with open('../data/LONG-SPEC2K6-00.res','r') as f:
        for index,line in enumerate(tqdm(f.readlines())):
        # for line in tqdm(f.readlines()):
            line=line.strip()
            if line=='':
                continue
            pc=int(line.split(' ')[0])
            direction=int(line.split(' ')[1])
            pht_index=(ghr^pc)%(2**17)
            pht_counter=pht[pht_index]
            if pht_counter>1:
                predict_direction=1
            else:
                predict_direction=0
            total_count+=1
            if direction==predict_direction:
                right_count+=1
            if direction==1:
                pht[pht_index]=min(3,pht_counter+1)
            else:
                pht[pht_index]=max(0,pht_counter-1)
            ghr=((ghr<<1)+direction)&max_ghr
            if index%100000==0:
                print('accuracy:',right_count/total_count)

if __name__ == '__main__':
    main()
    print('All is well!')

# NUM_INSTRUCTIONS     	 :  149970336
# NUM_CONDITIONAL_BR   	 :   25181955
# NUM_MISPREDICTIONS   	 :     596047
# MISPRED_PER_1K_INST  	 :      3.974