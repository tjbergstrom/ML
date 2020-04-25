# Ty Bergstrom
# Monica Heim
# auto_test.py
# CSCE A415
# April 23, 2020
# ML Final Project
# automated testing with different parameters and hypertuning
#
# terminal command usage:
# open the virtual environment:
# $ source ./venv1/bin/activate
# and run with this command:
# $ python3 auto_test.py


import os

aug = ["-a original ",  "-a light1 ", "-a light2 ", "-a light3 ", "-a medium1 ", "-a medium2 ", "-a medium3 ", "-a heavy1 ", "-a heavy2 "]
opt = ["-o Adam ", "-o Adam2 ", "-o Adam3 ", "-o Adam4 ", "-o SGD ", "-o SGD2 ", "-o SGD3 ", "-o RMSprop ", "-o Adadelta "]
bs = ["-b xs ", "-b s ", "-b ms ", "-b m ", "-b lg ", "-b xlg "]
imgsz = ["-i xs ", "-i s ", "-i m ", "-i lg ", "-i xlg "]
models = ["-m lenet ", "-m cnn ", "-m vgg "]
epochs = ["-e 25 ", "-e 50 ", "-e 75 ", "-e 100 "]

plot = "-p image_processing/plot"
cmd = "python3 -W ignore train_a_model.py "
cmd += models[0]

#cmd += opt[3]
cmd += epochs[2]

i = 0
for o in opt[4:7] :
    arg1 = cmd + o
    for b in [ bs[2], bs[3] ] :
        plt = plot + str(i) + ".png "
        arg2 = arg1 + b + plt
        print("\n", arg2, "\n")
        os.system(arg2)
        i += 1

'''
i = 0
for o in [ opt[2], opt[7] ] :
    arg1 = cmd + o
    for b in bs:
        plt = plot + str(i) + ".png "
        arg2 = arg1 + b + plt
        print("\n", arg2, "\n")
        os.system(arg2)
        i += 1
'''

'''
i = 0
for im in [ imgsz[1], imgsz[2] ] :
    arg1 = cmd + im
    for e in [ epochs[0], epochs[1] ] :
        arg2 = arg1 + e
        for b in bs:
            plt = plot + str(i) + ".png "
            arg3 = arg2 + b + plt
            print("\n", arg3, "\n")
            os.system(arg3)
            i += 1
'''

'''
i = 0
for im in [ imgsz[1], imgsz[3], imgsz[4] ] :
    arg = cmd + im
    for e in [ epochs[1], epochs[2] ] :
        arg1 = arg + e
        for o in [ opt[0], opt[1], opt[2], opt[6], opt[7], opt[4] ] :
            arg2 = arg1 + o
            for b in [ bs[6], bs[7], bs[8] ] :
                plt = plot + str(i) + ".png "
                arg3 = arg2 + str(b) + plt
                print("\n", arg3, "\n")
                os.system(arg3)
                plt = plot
                i += 1
'''



#for o in opt:
    #out = cmd + o
    #os.system(out)
    #out = cmd
#for b in bs:
    #out = cmd + b
    #os.system(out)
    #out = cmd
#for a in aug:
    #out = cmd + a
    #os.system(out)
    #out = cmd
#for o in ["-o RMSprop ", "-o Adadelta2 "]:
    #out1 = cmd + o
    #for b in bs:
        #out2 = out1 + b
        #os.system(out2)
        #out2 = out1
    #out1 = cmd
#for a in aug[2:3]:
    #out = cmd + a
    #os.system(out)
    #out = cmd















