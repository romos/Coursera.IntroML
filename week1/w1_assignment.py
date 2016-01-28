import pandas

def readcsv(fin):
	return pandas.read_csv(fin,index_col='PassengerId')

def sex_count(data, fout_name):
	# # my submission:	
	# male = data[data['Sex'] == 'male']['Sex'].count()
	# female = data[data['Sex'] == 'female']['Sex'].count()
	# sexnum = {}
	# sexnum['male'] = male
	# sexnum['female'] = female
	# return sexnum
	
	sex = data['Sex'].value_counts()
	
	# Print out the results
	fout = open(fout_name, 'w')
	print ('%d %d' % (sex['male'], sex['female']), file = fout, end="")
	fout.close()	

def main():
	fin = 'titanic.csv'
	data = readcsv(fin)

	print ('Sex Count task...')
	sex_count(data, 'sex_count.txt')
	
	print ('Completed!')

if __name__ == '__main__':
	main()