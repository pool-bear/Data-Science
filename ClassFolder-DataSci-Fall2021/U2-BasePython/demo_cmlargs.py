"""
command line arguments
"""
import sys
print('sys.argv = ', sys.argv)

i = 0
for _ in sys.argv:
    print('sys.argv['+ str(i) + ']= ', sys.argv[i])
    i += 1





