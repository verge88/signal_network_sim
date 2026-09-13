import ctypes
print("awareness:", ctypes.c_int.in_dll(ctypes.windll.shcore, "?").value if False else
      ctypes.windll.shcore.GetProcessDpiAwareness(0, ctypes.byref(ctypes.c_int())) )
a = ctypes.c_int()
ctypes.windll.shcore.GetProcessDpiAwareness(None, ctypes.byref(a))
print("0=unaware 1=system 2=per-monitor:", a.value)
print("system dpi:", ctypes.windll.user32.GetDpiForSystem())
