# Core-only: aggressive target so the fitter optimises for speed; the report's
# Fmax Summary gives the achieved Fmax regardless of this period.
create_clock -name clk -period 4.000 [get_ports {clk}]
derive_clock_uncertainty
