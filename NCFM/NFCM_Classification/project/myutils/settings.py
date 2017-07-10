'''
    There some settings
'''

fishname_to_int = {
    'ALB': 0,
    'BET': 1,
    'DOL': 2,
    'LAG': 3,
    'NoF': 4,
    'OTHER': 5,
    'SHARK': 6,
    'YFT': 7,
}

int_to_fishname = dict((value, key) for key, value in fishname_to_int.iteritems())