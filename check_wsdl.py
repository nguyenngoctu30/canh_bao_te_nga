import sys, os
found = []
for p in sys.path:
    try:
        fp = os.path.join(p, 'wsdl', 'devicemgmt.wsdl')
        if os.path.exists(fp):
            found.append(fp)
    except Exception:
        pass
print('FOUND_WSDL:', found)
print('SYS_PATHS sample:', sys.path[:5])
