python smalltest_fixes.py > raw
grep p raw | cut -d" " -f2,3 > phenotypes
grep m raw | cut -d" " -f2,3 > mutations
