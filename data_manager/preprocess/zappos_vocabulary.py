fname = '/Users/susanaparis/Documents/Belgium/DeepFashion/data/fashion53k/vocab/vocab_zappos.txt'

with open(fname, 'r') as f:
    zappos_words = f.read().replace('-', ' ').replace('/', ' ').split('\n')

zv = [z.lower() for z in zappos_words]
ZAPPOS_VOCAB_TUPLES = [tuple(z.split()) for z in zv]

ZAPPOS_VOCAB_LIST = ['_'.join(z) for z in ZAPPOS_VOCAB_TUPLES]



# print len(ZAPPOS_VOCAB)  # 212 words

# ZAPPOS_NGRAMS = ['Plus Size',
#                  'Knee Length',
#                  'Tea Length',
#                  'Little Black Dress',
#                  'Wear to Work',
#                  'Mother of the Bride',
#                  'A - line',
#                  'High Waist',
#                  'Wrap Dress',
#                  'High Low',
#                  'Baby Doll',
#                  'One Shoulder',
#                  'Dropped Waist',
#                  'Cover Up',
#                  'Short Sleeves',
#                  '3 4 Sleeve',
#                  '3 4 Sleeves',
#                  'Long Sleeves',
#                  'V - neck',
#                  'Off The Shoulder',
#                  'Mock Turtleneck',
#                  '1 4 Zip',
#                  'Animal Print',
#                  'Faux Leather',
#                  'Action Sports',
#                  'Floral Print',
#                  'Horizontal Stripes',
#                  'Tie - Dye',
#                  'Vertical Stripes',
#                  'Animal Print',
#                  'Polka Dot',
#                  'Leopard Print',
#                  'Snake Print',
#                  'Pin Stripes',
#                  'Dip - Dyed',
#                  'Zebra Print',
#                  'Cheetah Print',
#                  'Tattoo Print',
#                  'Cut - Outs',
#                  'Faux Pockets'
#                  '3 4 length',
#                  'short sleeve',
#                  'mid thigh'
#                  ]
# ZAPPOS_NGRAMS = [z.lower().replace('-', '') for z in ZAPPOS_NGRAMS]
#
# ZAPPOS_NGRAMS = [tuple(z.split()) for z in ZAPPOS_NGRAMS]  # list of tuples
#
# print zappos_ngrams

for w in ZAPPOS_VOCAB_LIST:
    print w




