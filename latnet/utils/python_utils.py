
from termcolor import colored, cprint

def str2shape(string):
  shape = string.split('x')
  return map(int, shape)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def print_dict(name, dictionary, color):
  print_string = name + '\n'
  names = dictionary.keys()
  names.sort()
  for name in names:
    if type(dictionary[name]) is not list:
      print_element = '   ' + name + ':'
      print_element = print_element.ljust(28)
      if type(dictionary[name]) is float:
        print_element += str(round(dictionary[name], 5))
      else:
        print_element += str(dictionary[name])
      print_element += '\n'
      print_string += print_element
  print_string = colored(print_string, color)
  return print_string

def llist2list(llist):
  out_list = []
  for j in xrange(len(llist[0])):
    store_out_list = []
    for i in xrange(len(llist)):
      store_out_list.append(llist[i][j])
    out_list.append(store_out_list)
  return out_list







 
