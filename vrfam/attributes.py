"""The 40 binary CelebA / CelebA-HQ attributes, in the order of the official annotation file."""

CELEBA_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
    'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
    'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
    'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
]

# Friendlier names used in the papers.
ALIASES = {'Gender': 'Male', 'Age': 'Young', 'Smile': 'Smiling', 'Glasses': 'Eyeglasses'}


def attribute_index(name):
    name = ALIASES.get(name, name)
    if name not in CELEBA_ATTRIBUTES:
        raise KeyError(f'Unknown attribute "{name}". Choose from: {", ".join(CELEBA_ATTRIBUTES)}')
    return CELEBA_ATTRIBUTES.index(name)


def canonical_name(name):
    return CELEBA_ATTRIBUTES[attribute_index(name)]
