# Forward Check

N = 1 -- zero rows in cmat

N, Re, Nts/12 = 2, 50, `2**8`

> Fail with `2**7`

```
from dolfin_navier_scipy.data_output_utils import plot_outp_sig
plot_outp_sig("/scratch/tbd/dnsdata/drc50.01.0_4552841-16__ssd0.0t0.0180.0000Nts23040ab0.01.0A1e-05")
```
large wiggles only at end (t=130+)

gonna check `u=[1, 1]` (was `[1, -1]`) maybe this triggers the large modes 

long time simu

```
from dolfin_navier_scipy.data_output_utils import plot_outp_sig
plot_outp_sig("/scratch/tbd/dnsdata/drc50.01.0_4552841-16__ssd0.0t0.0600.0000Nts153600ab0.01.0A1e-05", notikz=True)
```
