large domain setup
---

RE, Nts = 40, 15000
kappa = 0.001

## Gramians
pymess, hinf = 1, 1
ttf = 0

### controller mats
ttf_npicardsteps = 12: FAIL
ttf_npicardsteps = 18: WORK
ttf_npicardsteps = 15: WORK

### controller mats
kappa = 0.01
ttf_npicardsteps = 15: FAIL

RE=60
---
RE, Nts = 60, 15000
kappa = 0.001

### MyMESS-LQG
ttf = 0: WORK
ttf = 20: WORK
ttf = 10: FAIL
ttf = 15: FAIL
ttf = 18: FAIL 
ttf = 19: FAIL (almost but fail)
ttf = 20: WORK! (slow decay but decay)
