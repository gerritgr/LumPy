import os, glob

for i, f in enumerate(sorted(glob.glob('model/*.model'))[::-1]):
    try:
        os.system('python3 evaluation.py '+f)
        os.system('cp LumpingLog.log LumpingLog{}.log'.format(i))
    except:
        os.system('cp LumpingLog.log LumpingLog{}.log'.format(i))
        pass


