from project.src.imagepreparation.imageprep import tc
from project.src.imagepreparation.parameter1 import batch_size

#total_tr, total_vd, total_tt = tc()
#epochs = 10

#print(total_tr)
#step_size
#for model purpose

def ss():

    train_b, valid_b, test_b = batch_size(20, 20, 10)
    total_tr, total_vd, total_tt = tc()

    train_steps = total_tr//train_b #TOTAL TRAIN SET LENGTH INTEGER DIVISION BY TRAIN BATCH SIZE
    valid_steps = total_vd //valid_b
    test_steps = total_tt //test_b
    #print(train_steps)

    return train_steps, valid_steps, test_steps




