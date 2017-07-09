import numpy
data1 = numpy.genfromtxt('data/train.csv',delimiter=',',dtype=float)
data1 = numpy.delete(data1,(0), axis=0)
y = data1[0:data1.shape[0],data1.shape[1]-1:data1.shape[1]]
data = numpy.ones((data1.shape[0],data1.shape[1]-2+1),dtype='f');
data[0:data1.shape[0],0:data1.shape[1]-2] = data1[0:data1.shape[0],1:data1.shape[1]-1]	
rows = data.shape[0];
columns = data.shape[1];
lamb = 1;
accuracy = 100000;
warray = numpy.ones((columns,4),dtype='f');
####data scaling
avg = numpy.mean(data,axis=0);
sd = numpy.std(data,axis=0);
data[:,0:columns-1] = (data[:,0:columns-1] - avg[0:columns-1] )/ sd[0:columns-1];
####regression loop
for i in range(4):   #i for the loop for three values of the p
	p=i/4+1.25;
	#print(p)
	for k in range(accuracy):
		yc = numpy.dot(data,warray[:,i:i+1]);
		totalcost = -1*numpy.dot(data.transpose(),(y-yc)) 
		totalcost = totalcost +lamb*p/2*(numpy.multiply(numpy.absolute(warray[:,i:i+1])**(p-1),numpy.sign(warray[:,i:i+1])))
		warray[:,i:i+1] = warray[:,i:i+1] - 0.0005*totalcost;
	#print(sum(numpy.absolute(y-numpy.dot(data,warray[:,i:i+1]))))
####test data
testdata = numpy.genfromtxt('data/test.csv',delimiter=',',dtype=float)
testdata = numpy.delete(testdata,(0), axis=0)
testdata = numpy.delete(testdata,(0), axis=1)
testdata = numpy.append(testdata,numpy.ones((testdata.shape[0],1)),axis=1)
testdata[:,0:columns-1] = (testdata[:,0:columns-1]-avg[0:columns-1])/ sd[0:columns-1]
####calculating by closed form of the L2 norm
graddesvalue = numpy.dot(testdata,warray[:,3:4])
normal = numpy.linalg.inv(numpy.dot(data.transpose(),data) + numpy.identity(14))
normal = numpy.dot(normal,data.transpose())
normal = numpy.dot(normal,y)
print(numpy.sum(normal-warray[:,3:4]))
####output for the first loop of L2 norm
outputfile = open("output.csv", "w")
outputfile.write("ID,MEDV"+"\n")
predictedvalue = numpy.dot(testdata,warray[:,3:4])
for i in range(testdata.shape[0]):
	outputfile.write(str(i)+","+str(predictedvalue[i,0])+"\n")
#### output for other norms
for k in range(3):	# k is for looping for diff values of norm
	outputfile = open("output_p"+str(k)+".csv", "w")
	outputfile.write("ID,MEDV"+"\n")
	predictedvalue = numpy.dot(testdata,warray[:,k:k+1])
	for i in range(testdata.shape[0]):
		outputfile.write(str(i)+","+str(predictedvalue[i,0])+"\n")