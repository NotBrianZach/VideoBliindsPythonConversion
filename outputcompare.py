import numpy as np
import scipy.io as sio

np.set_printoptions(threshold=np.nan)

content_file = sio.loadmat('./matlabout.mat')
matlabout = content_file['overallcosts']
print "matlab costs"
print matlabout.shape
print matlabout

content_file = sio.loadmat('./matlabvects.mat')
matlabvects = content_file['dct_motion_comp_diff']
#matlabvects = content_file['motion_vects16x16']
print "matlab vects"
print matlabvects.shape
print matlabvects[:,:,0]

#with open('pythonout', 'r') as content_file:
#    pythonout = np.load(content_file)
#    print "python costs"
#    print pythonout.shape
#    print pythonout

with open('pythonvects', 'r') as content_file:
    pythonvects = np.load(content_file)
    print "python vects"
    print pythonvects.shape
    print pythonvects[:,:,0]
print "finaloutputcomparison"
#print pythonout.reshape(4683,3)-matlabout
#print "costs"
#print pythonout.reshape(4863,3)-matlabout

print "vects 0"
print pythonvects[:,:,0]-matlabvects[:,:,0]
print "vects 1"
print pythonvects[:,:,1]-matlabvects[:,:,1]
print "vects 2"
print pythonvects[:,:,2]-matlabvects[:,:,2]













#f = open('finaloutputcomparison','w')
#f.write(np.array_str(pythonarray) + "\n" + np.array_str(matlabarray))
#f.write(np.array_str(pythonarray - matlabarray))
#f.write(np.array_str(pythoncosts.reshape(4683,3) - matlabcosts))

#k = open('pythonout','w')
#k.write(np.array_str(pythoncosts))

#g = open('matlabout','w')
#g.write(np.array_str(matlabarray))

#pythonarray = eval(pythonarray)
#if data is a string, it will convert matlab array into numpy array
#content_file = sio.loadmat('./matlaboverallcosts.mat')
##matlabarray = content_file['overallcosts']
#
#content_file = sio.loadmat('./matlaboverallcosts.mat')
#matlabcosts= content_file['motion_vects16x16']
##matlabcosts = np.load(matlabcostsarray)
#print "matlab costs"
#print matlabcosts.shape
##print matlabcosts
##with open('./matlaboverallcosts', 'r') as content_file:
##    matlabcostsarray = content_file['overallcosts']
##    matlabcosts = np.load(matlabcostsarray)
##    print "matlab costs"
##    print matlabcosts.shape
##    print matlabcosts
##    #while (pythoncosts.any()):
#
#with open('./pythonarray', 'r') as content_file:
#    pythonarray = np.load(content_file)
#
#with open('pythoncosts', 'r') as content_file:
#    pythoncosts = np.load(content_file)
#    #while (pythoncosts.any()):
#    print "python costs"
#    print pythoncosts.shape
#    #print pythoncosts
#
#
#print "python array shape"
#print pythonarray.shape
#print pythonarray
#
#
##print "python costs reshape"
##print pythoncosts.reshape(4683,3).shape
##print "python costs2"
##print pythoncosts2
#
##print "matlab array shape"
##print matlabarray.shape
##print "finaloutputcomparison shape"
##print pythoncosts.reshape(4683,3)-matlabcosts
#
#print "pythoncosts"
#print pythoncosts
#print "matlabcosts"
#print matlabcosts
#
#print "finaloutputcomparison"
##print pythoncosts.reshape(4683,3)-matlabcosts
#print pythonarray-matlabcosts
#
##f = open('finaloutputcomparison','w')
##f.write(np.array_str(pythonarray) + "\n" + np.array_str(matlabarray))
##f.write(np.array_str(pythonarray - matlabarray))
##f.write(np.array_str(pythoncosts.reshape(4683,3) - matlabcosts))
#
#k = open('pythonout','w')
#k.write(np.array_str(pythoncosts))
#
##g = open('matlabout','w')
##g.write(np.array_str(matlabarray))
#
##pythonarray = eval(pythonarray)
##if data is a string, it will convert matlab array into numpy array
