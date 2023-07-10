'''
Common User Library

'''

def MakeFolder(fn):
    import os
    if not os.path.exists(fn):
        os.makedirs(fn)