import pandas as pd # csv handler
import networkx as nx # graph package
import clusterutils
from timeit import default_timer as timer

def main():
	start = timer()
	clusterutils.cluster_month()
	end = timer()

	print('Time = {}s'.format(end - start))
	print('End of Program.')



if __name__ == '__main__':
	main()
