import bug_lib
import sys

bug_no = int(sys.argv[1])
if bug_no == -1:
    bug_lib.cover_then_inject_bugs([])
else:
    bug_lib.cover_then_inject_bugs([bug_no])