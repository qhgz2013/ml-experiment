import sys
class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        if msvcrt.kbhit():
            return str(msvcrt.getch(), encoding='utf-8')
        else:
            return b'\0'


getch = _GetchWindows()