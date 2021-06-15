import os
import numpy as np

gt_file = 'test.txt'
error = 0.1
sgd_hidden_score = 5.70
custom_score = 8.45

    
def get_list(file_name):
    file = open(file_name)
    try:
        text = file.read().split()
        text = [float(x) for x in text]
    finally:
        file.close()
    return text
    

def check_score(file_name,baseline):
    if not os.path.exists(file_name):
        return False
    ### compute score
    test_text = np.array(get_list(file_name))
    length = len(test_text)
    gt_text = np.array(get_list(gt_file)[0:length])
    score = np.mean(np.abs(test_text-gt_text))
    print('The loss for:' + file_name + ' is:', score)
    return score < (baseline + error)
        
if __name__ == '__main__':
    test_files = {"sgd_hidden.txt":sgd_hidden_score,
                  "custom.txt":custom_score}
    for test_file,score in test_files.items():
        print(test_file+":"+str(check_score(test_file,score)))
        
        
    