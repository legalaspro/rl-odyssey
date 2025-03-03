from .dmc_wrapper import DMCWrapper, make_dmc
from .register import make


__all__ = ["DMCWrapper", 'make_dmc']

# # Explicitly expose these attributes to the module level
# DMCWrapper = DMCWrapper
# make = make
# #https://github.com/jonzamora/dmc2gymnasium/tree/main
# # https://github.com/denisyarats/dmc2gym 