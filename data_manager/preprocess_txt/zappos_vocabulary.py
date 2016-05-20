fname = '../../data/fashion53k/vocab/vocab_zappos.txt'

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


# plus_size
# petite
# juniors
# short
# long
# knee_length
# tea_length
# asymmetrical
# casual
# dress
# evening
# cocktail
# outdoor
# wedding
# bridesmaid
# prom
# homecoming
# little_black_dress
# office
# career
# wear_to_work
# mother_of_the_bride
# athletic
# nightclub
# a_line
# tank
# sheath
# shift
# maxi
# high_waist
# gown
# strapless
# wrap_dress
# halter
# tunic
# high_low
# shirt
# empire
# baby_doll
# one_shoulder
# dropped_waist
# bustier
# cover_up
# sport
# sweater
# coat
# sleeveless
# short_sleeves
# 3_4_sleeve
# long_sleeves
# scoop
# v_neck
# crew
# boatneck
# sweetheart
# keyhole
# square_neck
# spread
# off_the_shoulder
# cowl
# point
# mock_turtleneck
# wide
# mandarin
# turtleneck
# banded
# ballet
# snap
# peter_pan
# notch_lapel
# shawl
# 1_4_zip
# black
# blue
# white
# multi
# pink
# navy
# gray
# green
# red
# orange
# purple
# beige
# yellow
# gold
# coral
# brown
# bone
# tan
# silver
# burgundy
# khaki
# neutral
# taupe
# animal_print
# olive
# metallic
# mahogany
# bronze
# pewter
# polyester
# spandex
# cotton
# rayon
# nylon
# viscose
# jersey
# lace
# modal
# chiffon
# silk
# mesh
# linen
# crochet
# lyocell
# lycra
# acetate
# wool
# ponte
# satin
# pique
# denim
# acrylic
# twill
# terry
# faux_leather
# leather
# ramie
# velvet
# taffeta
# hemp
# tweed
# canvas
# synthetic
# cashmere
# felt
# microfiber
# action_sports
# summer
# western
# spring
# resort
# street
# fall
# skate
# winter
# surf
# retro
# sports
# athleisure
# floral_print
# horizontal_stripes
# geometric
# tie_dye
# vertical_stripes
# lace
# paisley
# animal_print
# chevron
# abstract
# polka_dot
# metallic
# tropical
# jacquard
# ombre
# leopard_print
# plaid
# snake_print
# pin_stripes
# dip_dyed
# zebra_print
# brocade
# aztec
# reptile
# gingham
# camo
# checkered
# cheetah_print
# argyle
# tattoo_print
# houndstooth
# patchwork
# ruched
# pleated
# cut_outs
# logo
# scalloped
# ruffles
# beaded
# embroidered
# piping
# sequins
# smocked
# fringe
# bows
# applique
# zipper
# crystals
# screenprint
# contrast_stitching
# peplum
# rhinestones
# buttons
# chains
# studded
# flowers
# faux_pockets
# grommets
# epaulette
# ribbons
# rivets





