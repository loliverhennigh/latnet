
import sys
sys.path.append('../')
import sim_runner.que as que

#q = que.Que([0])
q = que.Que([0,1,2]) # if you have more then one gpu put it here
q.enque_file("experiments.txt")
q.start_que_runner()





