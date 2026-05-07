import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yang/order_exploration/src/exploration/.colcon/install/exploration'
