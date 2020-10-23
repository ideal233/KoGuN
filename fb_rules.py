template = 'Y VY D YTY YBY => F'
r1 = 'L PO ANY ANY ANY=>T' # act 1. L is Large, PO is positive
r2 = 'S ANY ANY ANY ANY=>N' # act 0. S is Small, NE is negative
# r2 = 'S NE ANY ANY ANY=>N' # act 0. S is Small, NE is negative
r3 = 'ANY PO ANY ANY PO=>T' # act 1
r4 = 'ANY NE ANY NE ANY=>N' # act 0


rules = [r1, r2, r3, r4]

def y_l(v):
    if v > 250:
        return 1.0
    elif v < 200:
        return 0
    else:
        return (1/50)*v-4.0

def y_s(v):
    if v > 200:
        return 0
    elif v < 150:
        return 1.0
    else:
        return -(1/50)*v+4

def vy_po(v):
    if v < 0:
        return 0.0
    elif v > 6.0:
        return 1.0
    else:
        return (1/6.0) * v

def vy_ne(v):
    if v > 0:
        return 0.0
    elif v < -6.0:
        return 1.0
    else:
        return -(1/6.0) * v

def yty_ne(v):
    if v < -30:
        return 1.0
    elif v > 0:
        return 0
    else:
        return -(1/30.0) * v

def yby_po(v):
    if v > 30:
        return 1.0
    elif v < 0:
        return 0
    else:
        return (1/30.0) * v

def any(v):
    return 1.0

def no(v):
    return 0.0

# # Main engine positive large
# def main_engine_pl(v):
#     if v < 0.0:
#         return 0.0
#     elif v >= 1.0:
#         return 1.0
#     else:
#         return  v
#
# def horizontal_engine_p(v):
#     if v > 1.0:
#         return 1.0
#     elif v < 0.5:
#         return 0
#     else:
#         return 2.0 * v -1.0
#
# def horizontal_engine_n(v):
#     if v < -1.0:
#         return 1.0
#     elif v > -0.5:
#         return 0.0
#     else:
#         return -2.0 * v - 1.0

def if_t(strength):
    raise NotImplementedError

def if_n(strength):
    raise NotImplementedError

# def if_m(strength):
#     raise NotImplementedError
#
# def if_n(strength):
#     raise NotImplementedError

membership_functions = {
    'YL':y_l,
    'YS':y_s,
    'YANY':any,
    'VYPO':vy_po,
    'VYNE':vy_ne,
    'VYANY':any,
    'DANY':any,
    'YTYANY':any,
    'YTYNE':yty_ne,
    'YBYANY':any,
    'YBYPO':yby_po,
    # 'FP': f_t,
    # 'FN': f_n
}
imfs = {
    'IFT': if_t,
    'IFN': if_n
}