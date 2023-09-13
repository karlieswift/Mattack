"""
@Env: /anaconda3/python3.10
@Time: 2023/4/29-15:54
@Auth: karlieswift
@File: myGlobal.py
@Desc: 
"""

def _init():
    global _global_dict
    _global_dict = {}


def set_value(key, value):
    _global_dict[key] = value


def get_value(key):
    try:
        return _global_dict[key]
    except:
        print('read' + key + 'failure\r\n')

