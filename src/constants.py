# X-fact constants
XFACT_LABELS = ['false',
 'partly true/misleading',
 'true',
 'mostly true',
 'mostly false',
 'complicated/hard to categorise', 
 'other']

XFACT_LABEL2IDX = {label:idx for idx, label in enumerate(XFACT_LABELS)}
XFACT_IDX2LABEL = {idx:label for idx, label in enumerate(XFACT_LABELS)}
LABELS_TO_REMOVE = ['complicated/hard to categorise', 'other']


LANG_CODE2COUNTRY = {'tr': 'Turkey',
 'ka': 'Georgia',
 'pt': 'Brazil',
 'id': 'Indonesia',
 'sr': 'Serbia',
 'it': 'Italy',
 'de': 'Germany',
 'ro': 'Romania',
 'ta': 'Sri Lanka',
 'pl': 'Poland',
 'hi': 'India',
 'ar': 'Middle East',
 'es': 'Spain',
 'ru': 'Russia',
 'mr': 'Indian state of Maharashtra',
 'sq': 'Albania',
 'gu': 'Indian state of Gujarat',
 'fr': 'France',
 'no': 'Norway',
 'si': 'Sri Lanka',
 'nl': 'Netherlands',
 'az': 'Azerbaijan',
 'bn': 'Bangladesh',
 'fa': 'Persia',
 'pa': 'Indian state of Punjab'}

MONOLINGUAL_LANGUAGES=['ar','fr','fa','ru','id','pt'] 